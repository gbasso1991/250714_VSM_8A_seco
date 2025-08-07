#%% VSM muestra seca - Pablo Tancredi 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score 
from mlognormfit import fit3
from mvshtools import mvshtools as mt
import re
from uncertainties import ufloat
#%%
def lineal(x,m,n):
    return m*x+n

def coercive_field(H, M):
    """
    Devuelve los valores de campo coercitivo (Hc) donde la magnetización M cruza por cero.
    
    Parámetros:
    - H: np.array, campo magnético (en A/m o kA/m)
    - M: np.array, magnetización (en emu/g)
    
    Retorna:
    - hc_values: list de valores Hc (puede haber más de uno si hay múltiples cruces por cero)
    """
    H = np.asarray(H)
    M = np.asarray(M)
    hc_values = []

    for i in range(len(M)-1):
        if M[i]*M[i+1] < 0:  # Cambio de signo indica cruce por cero
            # Interpolación lineal entre (H[i], M[i]) y (H[i+1], M[i+1])
            h1, h2 = H[i], H[i+1]
            m1, m2 = M[i], M[i+1]
            hc = h1 - m1 * (h2 - h1) / (m2 - m1)
            hc_values.append(hc)

    return hc_values
#%% Levanto Archivos
data_parafilm = np.loadtxt('Parafilm1.txt', skiprows=12)
data_8A = np.loadtxt('8A_seco2.txt', skiprows=12)

#%% Armo vectores
H_parafilm = data_parafilm[:, 0]  # Gauss
m_parafilm = data_parafilm[:, 1]  # emu

H_8A = data_8A[:, 0]  # Gauss
m_8A = data_8A[:, 1]  # emu

fig1, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
ax.plot(H_parafilm, m_parafilm, '.-', label='Parafilm')
ax.plot(H_8A, m_8A, '.-', label='8A seco')

for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu)')
plt.xlabel('H (G)')
plt.show()
#%% Resto contribucion lineal
# normalizo -> pendiente -> escaleo -> resto
masa_pfilm_virgen=56.20 #mg (medida sin NP)
masa_pfilm= 90.08 #mg (antes de depositar 50 uL de FF)
masa_pfilm_FF = 90.33 #mg (una vez secos los 50 uL) 
masa_NP_8A=(masa_pfilm_FF-masa_pfilm)*1e-3 #g


m_pfilm_norm = m_parafilm/(masa_pfilm_virgen*1e-3) #emu/g  -  Normalizo

(pend,ord),pcov=curve_fit(lineal,H_parafilm,m_pfilm_norm) # (emu/g , emu/g/G) - Ordenada/Pendiente 

susceptibilidad_parafilm=ufloat(pend,np.sqrt(np.diag(pcov))[0]) # emu/g/G
print(f'Susceptibilidad Parafilm: {susceptibilidad_parafilm:.1ue} emu/g/G')
m_aux=(ord + pend*H_8A)*(masa_pfilm*1e-3)   #emu - Escaleo
m_8A_sin_diamag=m_8A-m_aux #emu   - Resto

fig1, ax = plt.subplots(figsize=(6,4), constrained_layout=True)

ax.plot(H_parafilm, m_parafilm, '-', label='Parafilm')
ax.plot(H_8A, m_8A, '-', label='8A seco')
ax.plot(H_8A, m_8A_sin_diamag, '-', label='8A s/ diamag')

# ax.plot(H_8A, y_aux, '-', label='aux')
for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu)')
plt.xlabel('H (G)')

#%% Normalizo por masa de NP
 
m_8A_sin_diamag_norm=m_8A_sin_diamag/masa_NP_8A #emu/g


fig1, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
#ax.plot(H_pat, m_pat, '-', label='Patron')
ax.plot(H_8A, m_8A_sin_diamag_norm, '-', label='8A')

for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu/g)')
plt.xlabel('H (G)')
plt.show()

# %%
H_anhist_8A, m_anhist_8A = mt.anhysteretic(H_8A, m_8A_sin_diamag_norm)
fit_8A = fit3.session(H_anhist_8A, m_anhist_8A, fname='8A', divbymass=False)
fit_8A.fix('sig0')
fit_8A.fix('mu0')
fit_8A.free('dc')
fit_8A.fit()
fit_8A.update()
fit_8A.free('sig0')
fit_8A.free('mu0')
fit_8A.set_yE_as('sep')
fit_8A.fit()
fit_8A.update()
fit_8A.save()
fit_8A.print_pars()
H_8A_fit = fit_8A.X
m_8A_fit = fit_8A.Y
#%%

hc_vals_G = coercive_field(H_8A, m_8A_sin_diamag_norm)
hc_G=ufloat(np.mean(np.abs(hc_vals_G)),np.std(np.abs(hc_vals_G)))
print(f"Campo coercitivo: Hc = {hc_G} G = {hc_G/(4*np.pi)} kA/m")

fig1, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
ax.plot(H_8A, m_8A_sin_diamag_norm, '-', label='8A')
ax.plot(H_8A_fit, m_8A_fit, '-', label='8A fit ')

ax.text(0.25,.5,f'masa NP = {masa_NP_8A*1e3:.2f} mg',bbox=dict(alpha=0.7),va='center',ha='center',transform=ax.transAxes)
ax.text(0.75,.5,f'Hc = {hc_G/(4*np.pi) } kA/m\nMs = {max(m_8A_sin_diamag_norm):.2f} emu/g',bbox=dict(alpha=0.7),va='center',ha='center',transform=ax.transAxes)

for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu/g)')
plt.xlabel('H (G)')
plt.savefig('ciclo_VSM_8A.png',dpi=300)
plt.show()

