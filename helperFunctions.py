import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.constants import physical_constants
from scipy.optimize import minimize
from matplotlib import cm

# suppress warnings (labellines)
import warnings
warnings.filterwarnings("ignore")

# gyromagnetic ratio in [rad/s/T]
gamma_p = physical_constants['proton gyromag. ratio'][0]

# vacuum magnetic permeability in [N/A**2]
mu_0 = physical_constants['vacuum mag. permeability'][0]

# field conversion of the dressing field in [uT/Vpp]
fieldConversion = np.array([17.90142671025224, 0.07157374413471178])

#def besselFct (x, xs, a, o=0, order=0):
#    return a * jv(order, x/xs) + o

def gaussFct(f, f0, A, s, o=0):
    """
    Gaussian function, used to fit the Rabi resonances. 
    """
    return A * np.exp(-(f-f0)**2/(2*s**2)) + o

def multiGaussFct (f, *p):
    sig = np.zeros_like(f)
    for i in range(len(popt)//4):
        sig += gaussFct(f, *p[4*i:4*(i+1)])
    return sig

def twoGaussFct (f, *p):
    return gaussFct(f, *p[0:3]) + gaussFct(f, *p[3:6]) + p[6]

def threeGaussFct (f, *p):
    return gaussFct(f, *p[0:3]) + gaussFct(f, *p[3:6]) + gaussFct(f, *p[6:9]) + p[9]

def fourGaussFct (f, *p):
    return gaussFct(f, *p[0:3]) + gaussFct(f, *p[3:6]) + gaussFct(f, *p[6:9]) + gaussFct(f, *p[9:12]) + p[12]


def calcEnergyLevels_X (X, y, N=46, n=0):
    
    if (N-2) % 4 != 0:
        raise ValueError('the following condition has to be fulfilled: (N-2)%4==0')

    EV = np.zeros((len(X), N))
    
    for j,x in enumerate(X):

        # create the diagonal elements of H0
        h0_pos = (n + np.arange((N-2)/4, -N/4, -1, dtype=int)) + y/2
        h0_neg = (n + np.arange((N-2)/4, -N/4, -1, dtype=int)) - y/2
        diags = np.vstack((h0_pos,h0_neg)).flatten('F')

        # put the diagonal elements into the H0 matrix
        H0 = np.zeros((N,N))
        for i in range(N):
            H0[i,i] = diags[i]
        
        # create the interaction matrix
        off_1 = np.ones(N-1)/4
        off_1[0::2] = 0
        off_3 = np.ones(N-3)/4
        off_3[1::2] = 0
        Vx = (sp.sparse.diags(off_1, offsets=1).toarray()
              + sp.sparse.diags(off_3, offsets=3).toarray() 
              + sp.sparse.diags(off_1, offsets=-1).toarray() 
              + sp.sparse.diags(off_3, offsets=-3).toarray())
        Vx = x * Vx

        # create the full Hamiltonian matrix
        H = H0 + Vx

        # calculate the eigenvalues of the Hamiltonian
        eigvals = np.linalg.eigvalsh(H)
        EV[j] = eigvals

    return EV

def analyzeDressedState (path, i, **kwargs):
    
    # get kwargs if existing
    popt = kwargs.get('p0') if 'p0' in kwargs else (500, -100, 50, 1)
    rang = kwargs.get('rang') if 'rang' in kwargs else (-np.inf, np.inf)
    
    # load data
    data = np.load(path+'dressedStates_{:02d}.npz'.format(i))
    
    F_SF = data['F_SF']
    Amp = data['Amp']
    vd = data['Vd'][i]

    mask = (F_SF>=rang[0]) & (F_SF<=rang[1])

    popt, pcov = curve_fit(gaussFct, F_SF[mask], Amp[0][mask], sigma=Amp[1][mask], absolute_sigma=True, p0=popt, maxfev=100000)
    perr = np.sqrt(np.diag(pcov))
    chi2 = np.sum((Amp[0][mask]-gaussFct(F_SF[mask], *popt))**2 / Amp[1][mask]**2)
    chi2_r = chi2 / (len(Amp[0][mask]) - len(popt))

    print('{} of {} mVpp'.format(i,vd))
    print('({:.1f} +/- {:.1f}) Hz'.format(popt[0], perr[0]))
    
    signal = abs(popt[1]/popt[2]/np.sqrt(2*np.pi)), 1/np.sqrt(2*np.pi) * np.sqrt(perr[1]**2/popt[2]**2 + popt[1]**2*perr[2]**2/popt[2]**4)
    noise = Amp[1].mean()
    print('SNR: {:.1f}'.format(signal[0]/noise))
    print('Amp: {:.3f} +/- {:.3f} ({:.1f})'.format(signal[0], signal[1], signal[0]/signal[1]))

    
    if 'verbose' in kwargs and kwargs.get('verbose')>0:
        print('dressing amplitude {} of {} mVpp'.format(i,vd))
        print('resonance at {:.1f}({:.0f}) Hz'.format(popt[0], 1e1*perr[0]))
        print('resonance width {:.1f}({:.0f}) Hz'.format(abs(popt[2]), 1e1*perr[2]))
        print('reduced chi-squared: {:.2f}\n'.format(chi2_r))
        print('signal: {:.3f}'.format(signal[0]))
        print('noise: {:.3f}'.format(noise))
        print(Amp[0][mask].std())
        
    if 'verbose' in kwargs and kwargs.get('verbose')>1:
        fig, ax = plt.subplots()
        ax.errorbar(F_SF[mask], Amp[0][mask], Amp[1][mask], fmt='C1.')
        ax.plot(F_SF[mask], gaussFct(F_SF[mask], *popt), 'C2-', lw=1, zorder=9)
        ax.grid()
        plt.title('dressing amplitude {} mVpp'.format(vd)); plt.xlabel('frequency [Hz]'); plt.ylabel('normalized spin polarization')
        fig.set(dpi=100)
        plt.show()
        
    fig, ax = plt.subplots()
    ax.errorbar(F_SF, Amp[0], Amp[1], fmt='C0.')
    ax.errorbar(F_SF[mask], Amp[0][mask], Amp[1][mask], fmt='C1.')
    ax.plot(F_SF, gaussFct(F_SF, *popt), 'C2-', lw=1, zorder=9)
    ax.grid()
    plt.title('dressing amplitude {} mVpp'.format(vd)); plt.xlabel('frequency [Hz]'); plt.ylabel('normalized spin polarization')
    fig.set(dpi=100)
    plt.show()
    
    return popt, perr



def findScale (dressAmp, resFrq, f0, fd, s0, Nlevels, **kwargs):
    
    tol = kwargs.get('tol') if 'tol' in kwargs else 1e-9
    
    # dressing parameter y
    y = f0 / fd
    
    # number of energy levels to calculate
    # size of matrix to diagonalize
    N = 46
    
    def minFct (s):

        # dressing parameter x
        X = dressAmp / s

        # calculate energy levels
        EV_y0 = calcEnergyLevels_X(X, y, N)

        # select the n=0 levels to search for crossings
        EV1 = EV_y0[:,N//2]
        EV2 = EV_y0[:,N//2-1]

        # find the crossing indices and manually ad the begining and the end
        dE = EV1 - EV2
        idx_peaks = sp.signal.find_peaks(-dE)[0]
        idx_peaks = np.concatenate(([0],idx_peaks+1,[None]))

        # flip energies at crossing points
        EV1_t = np.zeros(len(X))
        EV2_t = np.zeros(len(X))
        for n in np.arange(0, N, 2):
            EV1 = EV_y0[:,n]
            EV2 = EV_y0[:,n+1]
            for i in range(len(idx_peaks)-1):
                if i%2==0:
                    EV1_t[idx_peaks[i]:idx_peaks[i+1]] = EV1[idx_peaks[i]:idx_peaks[i+1]]
                    EV2_t[idx_peaks[i]:idx_peaks[i+1]] = EV2[idx_peaks[i]:idx_peaks[i+1]]
                else:
                    EV1_t[idx_peaks[i]:idx_peaks[i+1]] = EV2[idx_peaks[i]:idx_peaks[i+1]]
                    EV2_t[idx_peaks[i]:idx_peaks[i+1]] = EV1[idx_peaks[i]:idx_peaks[i+1]]
            EV_y0[:,n] = EV1_t
            EV_y0[:,n+1] = EV2_t

        # calculate the energy differences
        dE = np.zeros((N//2, len(X)))
        for i in np.arange(N//2):
            dE[i] = EV_y0[:,N//2+i] -  EV_y0[:,N//2-i-1]

        res = 0
        for i,amp in enumerate(dressAmp):
            for j,frq in enumerate(resFrq[i]):
                res += min((abs(fd*dE[:Nlevels,i]) - frq)**2)

        return res
    
    # minimize
    res = minimize(minFct, s0, tol=tol, method='L-BFGS-B')
    
    return res



def plotEnergyTransitions(f0, fd, dressAmp, resFrq, X, y, scale, Nlevels, N=122, **kwargs):
    
    # get kwargs if existing
    label = kwargs.get('label') if 'label' in kwargs else ''
    
    # calculate the energy levels
    EV_y0 = calcEnergyLevels_X(X, y, N)
    
    # select the n=0 levels to search for crossings
    EV1 = EV_y0[:,N//2]
    EV2 = EV_y0[:,N//2-1]

    # find the crossing indices and manually ad the begining and the end
    dE = EV1 - EV2
    idx_peaks = sp.signal.find_peaks(-dE)[0]
    idx_peaks = np.concatenate(([0],idx_peaks+1,[None]))

    # flip energies at crossing points
    EV1_t = np.zeros(len(X))
    EV2_t = np.zeros(len(X))
    for n in np.arange(0, N, 2):
        EV1 = EV_y0[:,n]
        EV2 = EV_y0[:,n+1]
        for i in range(len(idx_peaks)-1):
            if i%2==0:
                EV1_t[idx_peaks[i]:idx_peaks[i+1]] = EV1[idx_peaks[i]:idx_peaks[i+1]]
                EV2_t[idx_peaks[i]:idx_peaks[i+1]] = EV2[idx_peaks[i]:idx_peaks[i+1]]
            else:
                EV1_t[idx_peaks[i]:idx_peaks[i+1]] = EV2[idx_peaks[i]:idx_peaks[i+1]]
                EV2_t[idx_peaks[i]:idx_peaks[i+1]] = EV1[idx_peaks[i]:idx_peaks[i+1]]
        EV_y0[:,n] = EV1_t
        EV_y0[:,n+1] = EV2_t

    # calculate the energy differences
    dE = np.zeros((N//2, len(X)))
    for i in np.arange(N//2):
        dE[i] = EV_y0[:,N//2+i] -  EV_y0[:,N//2-i-1]

    # flip sign according to |+> to |-> flip
    # this gives a positive resonance frequency for the main resonance at x=0
    sign = np.ones(Nlevels) if fd<f0 else -np.ones(Nlevels)
    sign[0::2] *= -1
    
    if 'verbose' in kwargs and kwargs.get('verbose')>0:
        fig, ax = plt.subplots()
        # plot energy levels
        for i in range(N):
            ax.plot(X, EV_y0[:,i], 'C{}-'.format(i%4), lw=1)
        # plot crossing points
        for x in X[np.array(idx_peaks[1:-1], dtype=int)-1]:
            ax.axvline(x, c='k', ls='--', lw=1)
        ax.set(xlabel='x', xlim=(X.min(), X.max()), ylim=(-2,2))
        plt.show()

    colors = cm.autumn(np.linspace(0,1,Nlevels))
    
    if 'ax' in kwargs:
        ax = kwargs.get('ax')
        # plot the data
        ax.errorbar(np.nan, np.nan, fmt='k.', ms=8, label=label)
        for i,amp in enumerate(dressAmp):
            for j,frq in enumerate(resFrq[i]):
                ax.errorbar(amp/scale[0], frq, fmt='k.', ms=8)
        # plot the calculated energy levels
        for i in np.arange(Nlevels):
            ax.plot(X, abs(fd*dE[i]*sign[i]), zorder=1, c=colors[i])
    else: 
        fig, ax = plt.subplots()
        # plot the data
        for i,amp in enumerate(dressAmp):
            for j,frq in enumerate(resFrq[i]):
                ax.errorbar(amp/scale[0], frq, fmt='k.', ms=8)
        # plot the calculated energy levels
        for i in np.arange(Nlevels):
            ax.plot(X, abs(fd*dE[i]*sign[i]), zorder=1)
        ax.set(title='$f_0$ = {} Hz  /  $f_d$ = {} Hz'.format(f0,fd), xlabel='$x$', ylabel='resonance frequency [Hz]')
        fig.set(dpi=100)
        fig.tight_layout()
        plt.show()



def plotEnergyDensity (path, **kwargs):
    
    files = sorted(glob(path+'dressedStates_*.npz'))
    
    F_SF = np.load(files[0])['F_SF']

    N = len(files)
    M = len(F_SF)

    signal = np.zeros((N,M))

    for n,file in enumerate(files):
        sig = np.load(file)['Amp'][0]
        
        if 'blc' in kwargs and kwargs.get('blc')==True:
            sigMean = sig[F_SF>2500].mean()
            signal[n] = sig-sigMean+1
        else:
            signal[n] = sig

    signal = np.flip(signal.T, axis=0)

    fig, ax = plt.subplots(figsize=(6,10))
    ax.imshow(signal, aspect=0.1, cmap='inferno') #RdBu, inferno

    xTicks = np.load(files[0])['Vd']
    nXLabels = 5
    stepX = int(N / (nXLabels - 1))
    xPositions = np.arange(0,N,stepX)
    xLabels = np.array(xTicks[::stepX],dtype=int)
    ax.set_xticks(xPositions)
    ax.set_xticklabels(xLabels, )
    ax.set_xlabel('amplitude [mVpp]')

    yTicks = F_SF
    nYLabels = 7
    stepY = int(M / (nYLabels - 1))
    yPositions = M-np.arange(0,M,stepY)
    yLabels = np.array(yTicks[::stepY],dtype=int)
    ax.set_yticks(yPositions)
    ax.set_yticklabels(yLabels)
    ax.set_ylabel('frequency [Hz]')

    fig.set(dpi=100)
    fig.tight_layout()

    plt.show()