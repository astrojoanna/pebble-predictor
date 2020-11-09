import numpy as np
from astropy import constants as c

def pebble_predictor(**pars):

    """
    returns estimates of pebble Stokes number St(t,r) and pebble flux Mdot_p(t,r) [g/s] at every location (rgrid) and time (tgrid) in your protoplanetary disk
    Use as St,Mdot_p=pebble_predictor(parameters) where the parameters that are needed to predict the flux are:
    rgrid:     radial grid [cm]
    tgrid:     time grid [s]
    Mstar:     mass of the central star [g]
    SigmaGas:  initial gas surface density at every point of rgrid [g/cm^2]
    SigmaDust: initial solids surface density at every point of rgrid [g/cm^2]
    T:         gas temperature at every point of rgrid [K]
    alpha:     turbulence strength parameter
    vfrag:     collisional fragmentation threshold velocity [cm/s]
    rhop:      bulk density of dust grains [g/cm^3]
        - keyword names should be identical to the list above
        - the order of keywords does not matter
    """

    rgrid = pars['rgrid']
    tgrid = pars['tgrid']
    Mstar = pars['Mstar']
    SigmaGas = pars['SigmaGas']
    SigmaDust = pars['SigmaDust']
    T = pars['T']
    alpha = pars['alpha']
    vfrag = pars['vfrag']
    rhop = pars['rhop']

    k_b = c.k_B.cgs.value
    m_p = c.m_p.cgs.value
    Grav = c.G.cgs.value
    AH2 = 2.e-15
    au = c.au.cgs.value

    a0 = 1.e-4 # starting size of dust [cm]
    fff = 30. # how many times growth needs to be faster than drift (see Okuzumi et al. )
    cs = np.sqrt(k_b*T/(2.3*m_p)) # sound speed
    OmegaK = np.sqrt(Grav*Mstar/rgrid**3.) # Keplerian frequency
    rhog = SigmaGas*OmegaK / (np.sqrt(2.*np.pi)*cs) # midplane gas density
    pg = rhog*cs**2. # gas pressure

    rInt = np.zeros(np.size(rgrid)+1) # grid cell interfaces estimate (needed for pressure gradient calculation)
    rInt[0] = 1.5*rgrid[0]-0.5*rgrid[1]
    rInt[1:-1] = 0.5*(rgrid[1:]+rgrid[:-1])
    rInt[-1] = 1.5*rgrid[-1] - 0.5*rgrid[-2]
    dr = rInt[1:] - rInt[:-1]
    pgInt = np.interp(rInt,rgrid,pg) # gas pressure at the interfaces
    eta = (pgInt[1:]-pgInt[:-1])/dr[:] /(2.*rhog*OmegaK**2.*rgrid) # pressure gradient

    stfrag = 0.37*vfrag**2./(3.*alpha*cs**2.) # Stokes number in turbulence-induced fragmentation regime
    stdf = 0.37*vfrag/(2.*abs(eta)*OmegaK*rgrid) # Stokes number in drift-induced fragmentation regime
    st0 = 0.5*np.pi*a0*rhop/SigmaGas # Stokes number of micron-sized monomers

    # prepare the matrices to fill
    st = np.zeros((np.size(tgrid),np.size(rgrid))) # final Stokes number prediction
    flux = np.zeros((np.size(tgrid),np.size(rgrid))) # final pebble flux prediction
    massout = np.zeros((np.size(tgrid),np.size(rgrid))) # solids mass outside of the cell (for the flux prediction)

    # initial condition:
    Z0 = SigmaDust/SigmaGas
    tgrowth = 1./((alpha/1.e-4)**0.3333*Z0*OmegaK*(rgrid/au)**(-0.3333)) # growth timescale
    for ir in range(0,np.size(rgrid)):
        massout[0,ir] = np.sum(2.*np.pi*rgrid[ir:]*dr[ir:]*SigmaDust[ir:]) # initial mass budget
    st[0,:] = st0
    vv = 2.*abs(eta)*OmegaK*rgrid*st[0,:]/(1.+st[0,:]**2.) # pebble drift velocity
    flux[0,:] = 2*np.pi*rgrid[:]*vv[:]*SigmaDust[:] # initial pebble flux

    # time "integration":
    for it in range(1,np.size(tgrid)):
        massout[it,:] = massout[it-1,:] - flux[it-1,:]*(tgrid[it]-tgrid[it-1]) # decrease the mass budget
        Z = Z0*(massout[it,:]/massout[0,:]) # Sigma_d/Sigma_g estimate
        stini = st0*np.exp(tgrid[it]/tgrowth[:])
        stdrift = Z/abs(eta)/fff # Stokes number in the drift dominated regime (depends on the solids mass budget as opposed to the fragmentation-dominated regime)
        st[it,:] = np.minimum(np.minimum(stfrag,stdf),stini) # initial growth vs fragmentation
        st[it,:] = np.minimum(st[it,:],stdrift) # vs radial drift
        st[it,:] = np.maximum(st[it,:],st0[:]) # Stokes number doesn't fall below its initial value
        vv = 2.*abs(eta)*OmegaK*rgrid*st[it,:]/(1.+st[it,:]**2.) # pebble drift velocity
        vv = np.minimum(vv[:],rgrid[:]/tgrowth[:]/fff) # flux restriction if the growth timescale is too long
        flux[it,:] = 2*np.pi*rgrid[:]*vv[:]*SigmaDust[:]*(massout[it,:]/massout[0,:]) # pebble flux (corrected by the remaining mass estimate)
    return st,flux
