import math
import numpy as np
from math import sqrt
from scipy.integrate import solve_ivp

def controle_pid(SP, PV, j, I_buffer, D_buffer, dt, Kp, Ti, Td, b, c, N, UU_bias, UU_min, UU_max, metodo="backward"):
    """Calcula a saída do controlador PID e seus parâmetros.

    Argumentos:
        SP (float): Setpoint da variável.
        PV (float): Valor da variável medida.
        j (int): Buffer para armazenar o passo de tempo contado em números inteiros.
        I_buffer: Buffer para armazenar os valores da ação integral.
        D_buffer: Buffer para armazenar os valores da ação proporcional.
        dt (float): Passo de tempo.
        Kp (float): Ganho proporcional do controlador.
        Ti (float): Ganho integral do controlador.
        Td (float): Ganho derivativo do controlador:
        b (float): Peso do setpoint na ação proporcional do controlador.
        c (float): Peso da derivada do setpoint na ação derivativa do controlador. 
        N (int): Fator de proporção da constante do filtro de 1ª ordem para reduzir o efeito do ruído e tornar o controlador causal.
        UU_bias (float): Valor base (ou bias) da ação de controle. Esse valor é somadao ao valor de saída Uop calculado pelo controlador 
            para determinar a ação de controle total do controlador.
        UU_min (float): Valor mínimo que a variável pode atingir.
        UU_max (float): Valor máximo que a variável pode atingir.
        metodo (string): Método para resolver o controlador PID. As opções são: Backward, Forward, Tustin e Ramp.

    Retorna:
        numpy.array(): Array do numpy contendo o valor de saída do PID e os parâmetros calculados.
    """
    # Define o método para o controlador PID:
    if metodo == "backward":
        b1 = Kp * dt / Ti if Ti != 0 else 0.0
        b2 = 0.0
        ad = Td / (Td + N * dt)
        bd = Kp * Td * N / (Td + N * dt)

    elif metodo == "forward":
        b1 = 0.0
        b2 = Kp * dt / Ti  if Ti != 0 else 0.0
        ad = 1 - N * dt / Td if Td != 0 else 0.0
        bd = Kp * N  

    elif metodo == "tustin":
        b1 = Kp * dt / 2 / Ti if Ti != 0 else 0.0
        b2 = b1
        ad = (2 * Td - N * dt) / (2 * Td + N * dt)
        bd = 2 * Kp * Td * N/(2 * Td + N * dt)   

    elif metodo == "ramp":
        b1 = Kp * dt / 2 / Ti if Ti != 0 else 0.0
        b2 = b1
        ad = np.exp(-N * dt / Td) if Td !=0 else 0.0
        bd = Kp * Td * (1 - ad) / dt
        
    # Ação derivativa:
    D = ad * D_buffer + bd * ((c * SP[j] - PV[j]) - (c * SP[j-1] - PV[j-1]))
    
    # Ação integral:
    II = b1 * (SP[j] - PV[j]) + b2 * (SP[j-1] - PV[j-1])
    I = I_buffer + II                         
   
    # Ação proporcional:
    P = Kp * (b * SP[j] - PV[j])
    
    # Saída do controlador PID:
    Uop = UU_bias + P + I + D

    # Implementa anti-reset windup:
    if Uop < UU_min:
        II = 0.0
        Uop = UU_min
    if Uop > UU_max:
        II = 0.0
        Uop = UU_max
    I = I_buffer + II
    
    # Retorna a saída e os parâmetros do controlador:
    return np.array([Uop, I, D])


def controle_boiler_on_off(Uop, medido, minimo, maximo):
    """Controle liga e desliga do boiler.

    Argumentos:
        Uop (int): Variável de controle do seletor do boiler (Sa).
        medido (float): Valor medido para a temperatura do boiler (Tq).
        minimo (float): Valor mínimo para a temperatura do boiler.
        maximo (float): Valor máximo para a temperatura do boiler.
    
    Retorna:
        Uop (int): Variável de controle do seletor do boiler (Sa).
    """
    if medido < minimo:
        Uop = 1
    elif medido > maximo:
        Uop = 0

    return Uop


def modelo_alimentacao_quente(xf, xq):
    """Modelo para a vazão válvula de alimentação da corrente quente (Fq).

    Argumentos:
        xf (float): Abertura da válvula de corrente fria.
        xq (float): Abertura da válvula de corrente quente.

    Retorna:
        Fq (float): Vazão da válvula da corrente quente.
    """
    t = 60 * (xq ** (8 / 5))
    n = -1 * (-180 - 45 * (xf ** 2) - 250 * (xq ** (16 / 5)) * (xf ** 2) - 1180 * (xq ** (16 / 5)) + 60 * sqrt(9 * (xq ** (16 / 5)) * (xf ** 2) + 50 * (xq ** (32 / 5)) * (xf ** 2)))
    d = 6552 * (xq ** (16 / 5)) * (xf ** 2) + 648 * (xf ** 2) + 16400 * (xq ** (32 / 5)) * (xf ** 2) + 900 * (xq ** (16 / 5)) * (xf ** 4) + 2500 * (xq ** (32 / 5)) * (xf ** 4) + 81 * (xf ** 4) + 55696 * (xq ** (32 / 5)) + 16992 * (xq ** (16 / 5)) + 1296

    Fq = t * sqrt(n / d)
    return Fq


def modelo_alimentacao_fria(xf, Fq):
    """Modelo para a vazão da válvula de alimentação da corrente fria (Ff).

    Argumentos:
        xf (float): Abertura da válvula de corrente fria.
        Fq (float): Vazão da válvula da corrente quente.

    Retorna:
        Ff (float): Abertura da válvula da corrente fria.
    """
    n = 2 * xf * sqrt(125 * xf * xf - Fq * Fq + 500) - xf * Fq
    d = (xf ** 2) + 4

    Ff = n / d
    return Ff


def modelo_valvula_saida(xs):
    """Modelo para a vazão válvula de saída (Fs).

    Argumentos:
        xs (float): Abertura da válvula de saída.

    Retorna:
        Fs (float): Vazão da válvula de saída.
    """
    n = 5 * (xs ** 3) * (sqrt(30)) * sqrt(-15 * (xs ** 6) + sqrt(6625 * (xs ** 12) + 640 * (xs ** 6) + 16))
    d = 20 * (xs ** 6) + 1

    Fs = n / d
    return Fs


def modelo_temperatura_boiler(t, Tq, Fq, Tf, Sa):
    """Modelo para a temperatura de aquecimento do boiler (Tq).

    Argumentos:
        t (float): Tempo para ser utilizado na solução da equação diferencial ordinária pelo integrador do scipy.
        Tq (float): Temperatura de aquecimento do boiler.
        Fq (float): Vazão da válvula do boiler.
        Tf (float): Temperatura da corrente fria.
        Sa (float): Seletor do aquecimento do boiler. 

    Retorna:
        Equação diferencial para o modelo do boiler.
    """   
    return Fq * (Tf - Tq) + 250 * Sa


def modelo_nivel_tanque(t, h, Ff, Fq, Fd, Fs):
    """Modelo para o nível do tanque (h).

    Argumentos:
        t (float): Tempo para ser utilizado na solução da equação diferencial ordinária pelo integrador do scipy.
        h (float): nível do tanque de aquecimento da mistura entre corrente quente e fria.
        Fq (float): Vazão da válvula da corrente quente.
        Fd (float): Vazão da válvula do distúrbio no tanque.
        Fs (float): Vazão da válvula de saída.

    Retorna:
        Equação diferencial para o modelo do nível do tanque.
    """       
    return (1 / 0.5) * (Ff + Fq + Fd - Fs)


def modelo_temperatura_tanque(t, Tt, Ff, Tf, Fq, Tq, Fd, Td, Sr, h):
    """Modelo para a temperatura de saída do tanque de aquecimento (Tt).

    Argumentos:
        t (float): Tempo para ser utilizado na solução da equação diferencial ordinária pelo integrador do scipy.
        Tt (float): Temperatura de saída do tanque de aquecimento da mistura entre corrente quente e fria.
        Ff (float): Vazão da válvula da corrente fria.
        Tf (float): Temperatura da corrente fria.
        Fq (float): Vazão da válvula da corrente quente.
        Tq (float): Temperatura de aquecimento do Boiler.
        Fd (float): Vazão da válvula do distúrbio no tanque.
        Td (float): Temperatura da corrente de distúrbio.
        Sr (float): Seletor da resistência elétrica do tanque de aquecimento.
        h (float): nível do tanque de aquecimento da mistura entre corrente quente e fria.

    Retorna:
        Equação diferencial para o modelo da temperatura de saída do tanque.
    """      
    return (1 / (0.5 * h)) * (Ff * (Tf - Tt) + Fq * (Tq - Tt) + Fd * (Td - Tt) + 80 * Sr)


def modelo_temperatura_saida(t, Ts, Tt, Fs, Tinf):
    """Modelo para a temperatura de saída do chuveiro (Ts).

    Argumentos:
        t (float): Tempo para ser utilizado na solução da equação diferencial ordinária pelo integrador do scipy.
        Ts (float): Temperatura de saída do chuveiro.
        Fs (float): Vazão da válvula de saída.
        Tinf (float): Temperatura ambiente.

    Retorna:
        Equação diferencial para o modelo da temperatura de saída do chuveiro.
    """      
    return (Fs / 2.5) * (Tt - Ts) - 0.8 * (((Tt - Tinf) * (Ts - Tinf) * (0.5 * (Tt - Tinf) + 0.5 * (Ts - Tinf))) ** (1 / 3))


def modelagem_sistema(t, Y, Sa, xf, xq, xs, Tf, Td, Tinf, Fd, Sr):
    """Reúne as equaçõs diferenciais para a modelagem do sistema com tempo morto.

    Argumentos:
        t (float): Tempo para ser utilizado na solução da equação diferencial ordinária pelo integrador do scipy.
        Y (numpy.array): Condições iniciais para nível do tanque de aquecimento (h), temperatura do boiler (Tq),
            temperatura de saída do tanque de aquecimento (Tt), temperatura de saída do chuveiro (Ts).
        xf (float): Abertura da válvula de corrente fria.
        Sa (float): Seletor do aquecimento do boiler.
        xq (float): Abertura da válvula de corrente quente.
        xs (float): Abertura da válvula de saída.
        Tf (float): Temperatura da corrente fria.
        Td (float): Temperatura da corrente de distúrbio.
        Tinf (float): Temperatura ambiente.
        Fd (float): Vazão da válvula do distúrbio no tanque.
        Sr (float): Seletor da resistência elétrica do tanque de aquecimento.

    Retorna:
        numpy.array(): Equações diferenciais ordinárias para serem resolvidas pelo integrador do scipy.
    """
    N   = 50
    Vt  = 5
    UAt = 0.01
    Fq = modelo_alimentacao_quente(xf, xq)
    Ff = modelo_alimentacao_fria(xf, Fq)
    Fs = modelo_valvula_saida(xs)
    
    Tq = Y[0]
    h  = Y[1]
    Tt = Y[2]
    #T = Y[3]
    
    EDOh = modelo_nivel_tanque(t, h, Ff, Fq, Fd, Fs)
    EDOTq = modelo_temperatura_boiler(t, Tq, Fq, Tf, Sa)
    EDOTt = modelo_temperatura_tanque(t, Tt, Ff, Tf, Fq, Tq, Fd, Td, Sr, h)
    # EDOTs = modelo_temperatura_saida(t, Ts, Tt, Fs, Tinf)

    return np.array([EDOTq, EDOh, EDOTt] + list((N - 1) / Vt * (Fs * (Y[2:N+1] - Y[3:N+2]) - UAt * (Y[3:N+2] - Tinf))))


def simulacao_malha_temperatura(SYS, Y0, UT, dt, I_buffer, D_buffer, Tinf, split_range): 
    """Modelagem do sistema com malha de inventário para nível do tanque, controle liga-desliga (on-off)
    para a temperatura do boiler (Tq) e malha cascata para temperatura de saída com split-range em Sr.

    Argumentos:
        SYS: Função contendo as equações diferenciais para a solução do problema.
        Y0 (numpy.array): Array contendo as condições inciais para nível do tanque de aquecimento (h), 
            temperatura do boiler (Tq), temperatura de saída do tanque de aquecimento (Tt), 
            temperatura de saída do chuveiro (Ts).
        UT (numpy.array): Array contendo a matriz de variáveis e perturbações.
        dt (float): Intervalo de tempo.
        I_buffer (float): Buffer para armazenar os valores da ação integral.
        D_buffer (float): Buffer para armazenar os valores da ação proporcional.
        Tinf (float): Temperatura ambiente, utilizada para definir valor de UU_bias para a temperatura de saída.
        split_range (int): Indica se deve ser utilizado o split-range ou não. Um valor de 0 indica que
            não será utilizado, enquanto que um valor de 1 indica que o split-range deve ser utilizado.
    
    Retorna:
        tuple: Tuple com intervalo de tempo, as variáveis após a solução do sistema e o buffer do ajuste PID.
    """
    # Instantes de tempo:
    tempo_inicial = UT[0][0] 
    tempo_final = UT[1][0]
    
    # Armazenamento dos dados resultantes da simulação:
    TT = np.arange(start=tempo_inicial, stop=tempo_final + dt, step=dt, dtype="float")
    nt = np.size(TT)
    nu = np.size(UT, 1)
    ny = 4

    # Número correspondente a cada malha de controle em relação a posição do vetor Y0:
    # 0 - malha boiler, 1 - malha nível, 2 - malha tanque, 3 - malha saída:
    id = [0, 1, 2, -1]

    # Matriz para armazenar valores das variáveis de saída para cada instante de tempo inicializando com os valores: 
    # Inicializada com valores relativos a condição incial Y0:
    YY = np.ones((nt, ny))@np.diag(Y0[id])
    SP = np.ones((nt, ny))@np.diag(Y0[id])
    
    # Matriz para armazenar valores das variáveis de entrada para cada instante de tempo:
    UU = np.zeros((nt, nu-1)) 
  
    # Parâmetros das malhas de controle:
    Kp = np.array([1, 0.3, 2.0, 0.51])
    Ti = np.array([1, 0.8, 2.75, 0.6])
    Td = np.array([1, 0, 0, 0.05])
    b = np.array([1, 1, 1, 0.8])
    c = np.array([0, 0, 0, 0.1])
    UU_bias = np.array([0, 0.25, 0.25, Tinf])
    UU_min  = np.array([0, 0.01, 0.01, 20])
    UU_max  = np.array([1, 0.99, 0.99, 40])
    N = 10

    # Definindo o split-range:
    if split_range == 1:
        UU_max[2] = 1.99
    
    # Executando a simulação:
    ii = 0

    for k in np.arange(nt - 1):
        if TT[k] >= UT[ii+1, 0]:
            ii = ii + 1

        # Inicializando os valores: 
        UU[k,:] = UT[ii, 1:nu] 
        
        # Controle liga e desliga do boiler: malha 0
        SP[k, 0] = UU[k, 0]
        UU[k, 0] = controle_boiler_on_off(UU[k-1, 0], YY[k, 0], SP[k, 0] - 1, SP[k, 0] + 1)
        
        # Malhas de controle PIDs:
        for jj in [1, 2]:

            # Malha cascata:
            if jj == 2:
                # Setpoint da malha externa:
                SP[k, -1] = UU[k, jj]             
                uu = controle_pid(SP[:,-1], YY[:,-1], k, I_buffer[-1], D_buffer[-1], dt, Kp[-1], Ti[-1], Td[-1], b[-1], c[-1],
                                  N, UU_bias = UU_bias[-1], UU_min = UU_min[-1], UU_max = UU_max[-1])
                SP[k, jj] = uu[0]
                I_buffer[-1] = uu[1]
                D_buffer[-1] = uu[2]
            else:
                SP[k, jj] = UU[k, jj]

            uu = controle_pid(SP[:,jj], YY[:,jj], k, I_buffer[jj], D_buffer[jj], dt, Kp[jj], Ti[jj], Td[jj], b[jj], c[jj],
                              N, UU_bias = UU_bias[jj], UU_min = UU_min[jj], UU_max = UU_max[jj])

            # Split-range para controlar a temperatura do tanque:
            if jj == 2 and uu[0] > 1:
                UU[k, jj] = 1
                UU[k, -1] = uu[0] - 1
            else:
                UU[k, jj]  = uu[0]
                
            I_buffer[jj] = uu[1]
            D_buffer[jj] = uu[2]       
        
        # Integração do sistema:
        sol = solve_ivp(SYS, [TT[k], TT[k+1]], Y0, args=tuple(UU[k]), atol = 1e-8, rtol = 1e-8) 
        
        # Armazenamento dos valores calculados:
        Y0 = sol.y[:,-1]
        YY[k+1,:] = sol.y[id, -1]
                
    UU[k+1,:] = UU[k,:]  
    SP[k+1,:] = SP[k,:]
    
    return (TT, YY, UU, Y0, I_buffer, D_buffer)


def calculo_iqb(Ts, Fs):
    """Calcula o índice de qualidade do banho (iqb).
    
    Argumentos:
        Ts (float): Temperatura de saída do chuveiro.
        Fs (float): Vazão da válvula de saída.

    Retorna:
        iqb (float): Índice de qualidade do banho.
    """
    iqb = (1 / math.e) * math.exp((1 - ((Ts - 38 + 0.02 * Fs) / 2) ** 2) * np.power((0.506 + math.log10(math.log10((10000 * np.sqrt(Fs)) / (10 + Fs + 0.004 * np.power(Fs, 4))))), 20))

    if np.isnan(iqb) or iqb == None or np.isinf(abs(iqb)) or iqb < 0:
        iqb = 0
    if iqb > 1:
        iqb = 1
        
    return iqb


def custo_eletrico_banho(Sr, potencia_eletrica, custo_eletrico_kwh, tempo):
    """Calcula o custo da parte elétrica do banho.

    O custo da parte elétrica do banho é dado pela potência do chuveiro em KW multiplicado pela fração de utilização
    da resistência elétrica Sr, o custo do kWh em reais, e o tempo do banho em horas. Como o tempo é em minutos, 
    divide-se por 60. Como Sr é uma ação, seu valor é constante para toda a iteração.

    Argumentos:
        Sr (float): Seletor da resistência elétrica do tanque de aquecimento.
        potencial_eletrica (float): Potência elétrica do tanque de aquecimento (chuveiro) em kW.
        custo_eletrico_kwh (float): Custo do kWh da energia em reais por hora.
        tempo (float): Tempo da ação em minutos.

    Retorna:
        custo_eletrico_total (float): Custo da energia elétrica do banho em reais.
    """
    custo_eletrico_total = potencia_eletrica * Sr * custo_eletrico_kwh * tempo / 60

    return custo_eletrico_total


def custo_gas_banho(Sa, potencia_aquecedor, custo_gas_kg, dt):
    """Calcula o custo do gás do banho.

    A potência de um aquecedor a gás é dada em kcal/h. Considerando um rendimento de 86%, a potência últi será
    a potência multiplicada pelo rendimento. Para saber quantas kcal são fornecidas durante o banho, multiplica-se
    a potência útil pela quantidade de energia gasta de Sa (área da curva) e divide por 60 para tempo em horas.
    O pode calorífico do gás GLP é de 11750 kcal/kg. Se multiplicarmos esse valor pela quantidade de kcal gasta no banho,
    é possível obter a quantidade de gás em kg gasta no banho. Finalmente, o custo em reais é dado pela quantidade de gás em kg
    multiplicada pelo custo do gás em kg/reais.
    
    Referências: 
        https://conteudos.rinnai.com.br/vazao-aquecedor-de-agua-a-gas/#:~:text=A%20pot%C3%AAncia%20do%20aquecedor%20%C3%A9,hora%20(kcal%2Fh)
        https://www.supergasbras.com.br/super-blog/negocio/2021/qual-a-vantagem-do-poder-calorifico-do-glp#:~:text=GLP%3A%2011.750%20Kcal%2Fkg,G%C3%A1s%20Natural%3A%209.400%20Kcal%2Fm%C2%B3
    
    Argumentos:
        Sa (float): Seletor do aquecimento do boiler.
        potencial_aquecedor (float): Potência do aquecedor (boiler) em kcal/h.
        custo_gas_kg (float): Custo do kg do gás em reais por kg.
        tempo (float): Tempo da ação em minutos.
        df (float): Passo de tempo da simulação.

    Retorna:
        custo_gas_total (float): Custo do gás do banho em reais.
    """
    # Tempo em minutos que o aquecedor fica ligado (Sa = 1):
    # numero_vezes_aquecedor_ligado = np.count_nonzero(Sa == 1) # integrar também?
    # tempo_aquecedor_ligado = tempo * numero_vezes_aquecedor_ligado / (tempo / dt)

    # Quantidade de Sa utilizado:
    Sa_utilizado = np.trapz(y=Sa, dx=dt)

    # Potência útil do aquecedor em kcal/h:
    rendimento = 1
    potencia_util = potencia_aquecedor * rendimento

    # Quantidade de kcal fornecida durante o banho:
    # kcal_fornecida_no_banho = potencia_util * tempo_aquecedor_ligado / 60
    kcal_fornecida_no_banho = potencia_util * Sa_utilizado / 60

    # Poder calorífico do gás em kcal/kg (GLP):
    kg_equivalente_kcal = 11750 

    # Quantidade de gás gasta durante o banho:
    quantidade_gas_kg = kcal_fornecida_no_banho / kg_equivalente_kcal 

    # Custo do gás:
    custo_gas_total = custo_gas_kg * quantidade_gas_kg

    return custo_gas_total 


def custo_agua_banho(Fs, custo_agua_m3, tempo):
    """Calcula o custo de água do banho.

    A quantidade de litros gasta em um banho é dada pela vazão em L/min multiplicada pelo tempo em minutos do banho.
    Depois, divide-se a quantidade de litros por 1000, para obter a quantidade em m3 gasta durante o banho.
    Multiplicando esse valor pelo custo da água em m3/reais, obtém-se o custo em reais da água gasta durante o banho.

    Argumentos:
        xs (float): Vazão da válvula de saída.
        custo_agua_m3 (float): Custo da água em m3 por reais. 
        tempo (float): Tempo da ação em minutos.

    Retorna:
        custo_agua_total: Custo da água do banho em reais.
    """
    # integrar a vazão da água
    custo_agua_total = ((Fs * tempo ) / 1000) * custo_agua_m3 

    return custo_agua_total