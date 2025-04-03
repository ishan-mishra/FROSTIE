# module for the Hapke reflectance model

import numpy as np
import warnings
from .utils import spectra_list_match

class regolith:
    """
    Class to set up a 'regolith' made of one or more components and calculate it's reflectance.
    """
    def __init__(self):
        self.components = []
        self.constant_D = True

    def add_components(self, component_list, matched_axes=False):
        for component in component_list:
            self.components.append(component)
        
        if len(component_list) > 1:
            self.matched_axes = matched_axes

    def set_obs_geometry(self, i, e, g):
        self.i = i
        self.mu_0 = np.cos(np.deg2rad(self.i))
        self.e = e
        self.mu = np.cos(np.deg2rad(self.e))
        self.g = g
    
    # use property setter to update mu and mu_0 when i and e update, and vice versa

    @property
    def i(self):
        return self._i
    
    @i.setter
    def i(self, val):
        self._i = val
        self._mu_0 = np.cos(np.deg2rad(self._i))

    @property
    def mu_0(self):
        return self._mu_0
    
    @mu_0.setter
    def mu_0(self, val):
        self.i = np.rad2deg(np.arccos(val))

    @property
    def e(self):
        return self._e
    
    @e.setter
    def e(self, val):
        self._e = val
        self._mu = np.cos(np.deg2rad(self._e))

    @property
    def mu(self):
        return self._mu
    
    @mu.setter
    def mu(self, val):
        self.e = np.rad2deg(np.arccos(val))

    def set_porosity(self,p):
        if p < 0.48:
            raise ValueError('Porosity value needs to be higher than 1 for Hapke model to be valid')
        else:
            self.porosity = p

    # use property setter to update phi and K when porosity is updated

    @property
    def porosity(self):
        return self._porosity
    
    @porosity.setter
    def porosity(self, val):
        if val < 0.48:
            raise ValueError('Porosity value needs to be higher than 1 for Hapke model to be valid')
        else:
            self._porosity = val
            self._phi = 1 - self._porosity
            self._K = -np.log(1-1.209*self._phi**(2/3))/(1.209*self._phi**(2/3))

    @property
    def phi(self):
        return self._phi
    
    @property
    def K(self):
        return self._K

    def set_backscattering(self,B):
        self.B = B

    def set_s(self,s):
        self.s = s

    def set_mixing_mode(self, mixing_mode):
        if mixing_mode == 'intimate' or mixing_mode == 'linear':
            self.mixing_mode = mixing_mode
        else:
            raise ValueError("Mixing model type should be 'linear' or 'intimate'")

    
    def _calculate_w(self, n,k,wav,D,s,Q_E=1.0):
        return get_w(n,k,wav,D,s,Q_E=1.0, constant_D=self.constant_D)

    def _calculate_b(self,n,k,wav,D,s):
        return get_b(n,k,wav,D,s, constant_D=self.constant_D)

    def _calculate_c(self,n,k,wav,D,s):
        return get_c(n,k,wav,D,s, constant_D=self.constant_D)

    def _calculate_p(self, g,type='HG2',**kwags):
        return get_p(g,type='HG2',**kwags)

    def _calculate_r_one_comp(self):
        comp = self.components[0]
        self.b = self._calculate_b(comp['n'],comp['k'],comp['wav'],comp['D'],self.s)
        self.c = self._calculate_c(comp['n'],comp['k'],comp['wav'],comp['D'],self.s)
        self.p = self._calculate_p(self.g,type=comp['p_type'],b=self.b,c=self.c)
        self.w = self._calculate_w(comp['n'],comp['k'],comp['wav'],comp['D'],self.s)        
        return get_r(self.w,self.mu,self.mu_0,self.B,self.p,self.K)

    def calculate_reflectance(self, constant_D=True):
        if len(self.components) == 0:
            raise ValueError('The regolith is empty! Add components to it.')
        elif len(self.components) == 1:
            comp = self.components[0]
            self.model = self._calculate_r_one_comp()
            self.wav_model = comp['wav']

        elif len(self.components) > 1:
            
            if self.mixing_mode == 'linear':

                # calculate reflectance for each component

                self.b_list = []
                self.c_list = []
                self.p_list = []
                self.w_list = []

                self.r_list = []
                self.wav_r_list = []

                for comp in self.components:
                    b = self._calculate_b(comp['n'],comp['k'],comp['wav'],comp['D'],self.s)
                    self.b_list.append(b)
                    c = self._calculate_c(comp['n'],comp['k'],comp['wav'],comp['D'],self.s)
                    self.c_list.append(c)
                    p = self._calculate_p(self.g,type=comp['p_type'],b=b,c=c)
                    self.p_list.append(p)
                    w = self._calculate_w(comp['n'],comp['k'],comp['wav'],comp['D'],self.s)
                    self.w_list.append(w)        
                    self.r_list.append(get_r(w,self.mu,self.mu_0,self.B,p,self.K))
                    self.wav_r_list.append(comp['wav'])
                                
                # reduce all arrays to a common wavelength axis if needed
                
                if self.matched_axes:
                    
                    model_list, wav_model = self.r_list, self.wav_r_list[0]
                    
                else:
                
                    model_list, wav_model = spectra_list_match(self.r_list,self.wav_r_list)
                
                # calculate weighted average of all reflectances
                
                model_avg = 0
                
                for i,comp in enumerate(self.components):
                    model_avg += comp['f']*model_list[i]
                    
                
                self.model = model_avg
                self.wav_model = wav_model

            elif self.mixing_mode == 'intimate':
                
                # convert component parameters to lists for easier calculations
    
                n_list = []
                k_list = []
                wav_list = []
                f_list = []
                D_list = []
                p_type_list = []
                
                for comp in self.components:
                    n_list.append(comp['n'])
                    k_list.append(comp['k'])
                    wav_list.append(comp['wav'])
                    f_list.append(comp['f'])
                    D_list.append(comp['D'])
                    p_type_list.append(comp['p_type'])

                # reduce all n,k and wav arrays to a common wavelength axis if needed
    
                if self.matched_axes:
                    
                    wav_common = np.copy(wav_list[0])    # all wavelength arrays are the same
                    
                else:
                
                    n_list, wav_common = spectra_list_match(n_list,wav_list)
                
                    # returned wavelength array will be same as wav_common above
                    k_list, junk = spectra_list_match(k_list,wav_list) 

                # calculate w_mix and p_mix
    
                self.w_list, self.w_mix = get_w_mix(n_list,k_list,wav_common,D_list,f_list,self.s, constant_D=self.constant_D, include_D=True)
                
                self.p_list, self.p_mix = get_p_mix(n_list, k_list,wav_common,D_list,f_list,p_type_list,g=self.g,s=self.s, constant_D=self.constant_D)
                
                # calculate reflectance
                
                self.model = get_r(self.w_mix,self.mu,self.mu_0,self.B,self.p_mix,self.K)
                
                self.wav_model = wav_common
                



def get_w(n,k,wav,D,s, constant_D, Q_E=1.0):
    """Function to calculate the single scattering albedo, w for a particle. 
    Based on Bruce Hapke's model as implemented in his 2012 book.
    
    NOTE: We are assuming that diffraction is negligible so that Q_S = Q_s


    Parameters
    ----------
    n: 1D numpy float array
        spectrum of n, the real part of refractive index
    k: 1D numpy float array
        spectrum of k, the imaginary part of refractive index
    wav: 1D numpy float array
        the common wavelength axis for n and k in microns.
    D: float
        mean particle diameter in microns
    s: float
        internal scattering coefficient inside the particle
    Q_E: float
        volume-average extinction efficiency
    constant_D:Boolean
        if False, get_mrp function is used
        
    Unused parameters removed for now:    
    alt_defn: Boolean
        if True, alternate definition for S_e and S_i will be used (eg. Lapotre et al. (2017))
    mrp_spectrum: Boolean
        flag to calculate mean ray path length spectrum instead of a constant
        value

    Returns
    -------
    w: 1D numpy float array
        spectrum of the single scattering albedo

    """
    
    Q_S = get_Q_s(n,k,wav,D,s, constant_D=constant_D)  # volume-average scattering efficiency 
    
    w = Q_S/Q_E    # single-scattering albedo
    
    return w


def get_b(n,k,wav,D,s, constant_D):
    """Function to calculate the b parameter for the two-parameter Henyey-Greenstein function, in the 
    equivalent-slab approximation. Based on Bruce Hapke's model as implemented 
    in his 2012 book.
   
    Parameters
    ----------
    n: 1D numpy float array
        spectrum of n, the real part of refractive index
    k: 1D numpy float array
        spectrum of k, the imaginary part of refractive index
    wav: 1D numpy float array
        the common wavelength axis for n and k in microns.
    D: float
        mean particle diameter in microns
    s: float
        internal scattering coefficient inside the particle
    constant_D:Boolean
        if False, get_mrp function is used
        
    Returns
    -------
    b: 1D numpy float array
        spectrum of the parameter b

    """    
    
    del_Q_s = get_del_Q_s(n,k,wav,D,s, constant_D)
    Q_s = get_Q_s(n,k,wav,D,s, constant_D)
    
    b = 0.15 + 0.05/(1 + del_Q_s/Q_s)**(4/3)
    return b


def get_c(n,k,wav,D,s, constant_D):
    """Function to calculate the c parameter for the two-parameter Henyey-Greenstein function, in the 
    equivalent-slab approximation. Based on Bruce Hapke's model as implemented 
    in his 2012 book.
   
    Parameters
    ----------
    n: 1D numpy float array
        spectrum of n, the real part of refractive index
    k: 1D numpy float array
        spectrum of k, the imaginary part of refractive index
    wav: 1D numpy float array
        the common wavelength axis for n and k in microns.
    D: float
        mean particle diameter in microns
    s: float
        internal scattering coefficient inside the particle
    constant_D:Boolean
        if False, get_mrp function is used
        
    Returns
    -------
    c: 1D numpy float array
        spectrum of the parameter c

    """    
    
    del_Q_s = get_del_Q_s(n,k,wav,D,s, constant_D)
    Q_s = get_Q_s(n,k,wav,D,s, constant_D)
    
    c = del_Q_s/Q_s
    return c


def get_p(g,type='HG2',**kwags):
    """Function to calculate the phase function value. The
    Henyey-Greenstien functions are from Pg. 104 of Hapke 2012.

    Parameters
    ----------
    g: float
        phase angle in degrees
    type: str
        type of phase function to use. Options are 'isotropic','Euler', 'HG1' (one parameter Henyey-Greenstein)
        and 'HG2'
    **kwargs: additional arguments. 
        xi: float 
            the cosine asymmetry factor for type='HG1'
        b: 1D numpy float array
            the cosine asymmetry factor for type='HG2'
        c: 1D numpy float array
            backscattering fraction for type='HG2'
    """
    if type=='isotropic':
        return 1
    elif type=='Euler':
        return 1 - np.cos(np.deg2rad(g))
    elif type=='HG1':
        xi = kwags['xi']
        if xi <-1 or xi>1:
            raise ValueError('The value of xi should be between -1 and 1')
        return (1 - xi**2)/(1 + 2*xi*np.cos(np.deg2rad(g)) + xi**2)**(3/2)
    elif type=='HG2':
        b = kwags['b']
        c = kwags['c']
        if np.all(b <0) or np.all(b > 1):
            raise ValueError('The value of b should lie between 0 and 1')
        P = ((1 + c)/2)*(1 - b**2)/(1 - 2*b*np.cos(np.deg2rad(g)) + b**2)**(3/2) + \
    ((1 - c)/2)*(1 - b**2)/(1 + 2*b*np.cos(np.deg2rad(g)) + b**2)**(3/2)
        if np.all(P) < 0:
            raise ValueError('Phase function cannot be negative. Check the values of c')
        return P
    

def get_r(w,mu,mu_0,B,P,K=1.0):
    """Function to calculate reflectance (radiance factor or I/F) as given by Hapke's model 
    (Eq. 37 in Hapke 1981), which is also used by Lapotre et. al. 2017.

    Parameters
    ----------
    w: 1D numpy float array
        average particle's single scattering albedo spectrum 
    mu: float
        cosine of emergence angle
    mu_0: float
        cosine of incidence angle
    B: float
        backscattering function value
    P: float
        phase function value
    H: 1D numpy float array
        Chandrasekhar function value
    K: float
        porosity coefficient (see eqn. 8.70 in Hapke 2012)

    Returns
    -------
    r: 1D numpy float array
        reflectance or radiance factor spectrum
    """
    
    r = K*w*mu_0/(4*(mu_0+mu))*((1 + B)*P + get_H(w,mu/K)*get_H(w,mu_0/K) - 1)
    
    return r


def get_H(w,x):
    """Function to calculate Ambartsumian-Chandrasekhar H function (eq 8.56 from the book)

    Parameters
    ----------
    w: 1D numpy float array
        single scattering albedo spectrum
    x: float
        generic input variable, usually cosine of an angle

    Returns
    -------
    1D numpy array
        H function value
    """
    if x == 0 or x <0:
        warnings.warn('Since x is close to or less than 0, H is set to be 1 (Pg. 202 of Hapke 2012)')
        return np.ones(w.size)
    else:
        r_0 = (1 - np.sqrt(1-w))/(1 + np.sqrt(1-w))  #bihemispherical reflectance for isotropic
                                                 #scatterers
        temp = (1 - w*x*(r_0 + (1-2*r_0*x)/2*np.log((1+x)/x)))
    
        return 1/temp
    

def get_Q_s(n,k,wav,D,s, constant_D):
    """Function to calculate the volume scattering efficiency (ignoring diffraction) in the 
    equivalent-slab approximation. Based on Bruce Hapke's model as implemented in his 2012 book.

    Parameters
    ----------
    n: 1D numpy float array
        spectrum of n, the real part of refractive index
    k: 1D numpy float array
        spectrum of k, the imaginary part of refractive index
    wav: 1D numpy float array
        the common wavelength axis for n and k in microns.
    D: float
        mean particle diameter in microns
    s: float
        internal scattering coefficient inside the particle
    constant_D:Boolean
        if False, get_mrp function is used
        
    Unused parameters removed for now:
    
    mrp_spectrum: Boolean
        flag to calculate mean ray path length spectrum instead of a constant
        value
    alt_defn: Boolean
        if True, alternate definition for S_e and S_i will be used (eg. Lapotre et al. (2017))

    Returns
    -------
    Q_s: 1D numpy float array
        spectrum of the volume scattering efficiency

    """
    
    R_0 = ((n-1)**2 + k**2)/((n+1)**2 + k**2)    # normal specular reflection coefficient
    
    #if alt_defn:
    #    S_e = R_0 + 0.05 # surface external reflection coefficient
    #    S_i = 1.014 - 4./(n*(n+1)**2) # surface internal reflection coefficient
    
    S_e = 0.0587 + 0.8543*R_0 + 0.0870*R_0**2    
    S_i = 1 - 1/n*(0.9413 - 0.8543*R_0 - 0.0870*R_0**2)    
    
    alpha = 4*np.pi*k/wav   #  internal absorption coefficient
    
    r_i = (1-np.sqrt(alpha/(alpha+s)))/(1+(np.sqrt(alpha/(alpha+s)))) # internal diffusive bihemispherical reflectance
    
    if constant_D:
        mrp = 0.9*D    # true for n ~ 1.5-2.5 (Lucey (1998))
    else:
        mrp = get_mrp_spectrum(n,D)    # the mean ray path length through the interior 
    
    
    
    temp = np.sqrt(alpha*(alpha+s))
    
    Theta = (r_i + np.exp(-temp*mrp))/(1 + r_i*np.exp(-temp*mrp))  # particle internal transmission coefficient
    
    
    Q_s = S_e + (1-S_e)*(1-S_i)*Theta/(1-S_i*Theta)   # volume-average scattering efficiency 
    
    return Q_s

def get_del_Q_s(n,k,wav,D,s,constant_D):
    """Function to calculate the scattering efficiency difference in the 
    equivalent-slab approximation. Based on Bruce Hapke's model as implemented 
    in his 2012 book.

    Parameters
    ----------
    n: 1D numpy float array
        spectrum of n, the real part of refractive index
    k: 1D numpy float array
        spectrum of k, the imaginary part of refractive index
    wav: 1D numpy float array
        the common wavelength axis for n and k in microns.
    D: float
        mean particle diameter in microns
    s: float
        internal scattering coefficient inside the particle
    constant_D:Boolean
        if False, get_mrp function is used

    Returns
    -------
    del_Q_s: 1D numpy float array
        spectrum of the scattering efficiency difference

    """
    
    R_0 = ((n-1)**2 + k**2)/((n+1)**2 + k**2)    # normal specular reflection coefficient
    
    S_e = 0.0587 + 0.8543*R_0 + 0.0870*R_0**2    # surface external reflection coefficient
    
    S_i = 1 - 1/n*(0.9413 - 0.8543*R_0 - 0.0870*R_0**2)    # surface internal reflection coefficient
    
    alpha = 4*np.pi*k/wav   #  internal absorption coefficient
    
    r_i = (1-np.sqrt(alpha/(alpha+s)))/(1+(np.sqrt(alpha/(alpha+s)))) # internal diffusive bihemispherical reflectance
    
    if constant_D:
         mrp = 0.9*D    # true for n ~ 1.5-2.5 (Lucey (1998))
    else:
         mrp = get_mrp_spectrum(n,D)    # the mean ray path length through the interior
    
    temp = np.sqrt(alpha*(alpha+s))
    
    Psi = (r_i - np.exp(-temp*mrp))/(1 - r_i*np.exp(-temp*mrp))  # particle internal transmission coefficient
    
    
    del_Q_s = S_e + (1-S_e)*(1-S_i)*Psi/(1-S_i*Psi)   # volume-average scattering efficiency 
    
    return del_Q_s
    
def get_mrp_spectrum(n,D):
    """Function the calculate the mean ray path length spectrum through the interior of a slab.
    (equivalent slab approximation).
    
    Parameters
    ----------
    n: 1D numpy array
        the real part of the refractive index as a function of wavelength
    D: float
        mean particle diameter
    
    """
    mrp_spectrum= (2/3)*(n**2 - (1/n)*(n**2 - 1)**(3/2))*D
    return mrp_spectrum


def get_w_mix(n_list,k_list,wav_common,D_list,f_list,s, constant_D, include_D=True):
    """Function to calculate average single scattering albedo for a particulate mixture
    of two or more components. 
    
    NOTE: this function requires all n and k arrays to be at
    the resolution of wav_common (use utils.spectra_list_match function)

    Parameters
    ----------
    n_list: list of 1D numpy float arrays
        list of spectra of n, the real part of refractive index
    k_list: list of 1D numpy float arrays
        list of spectra of k, the imaginary part of refractive index
    wav_common: 1D numpy float array
        the common wavelength axis for all optical constants arrays
    D_list: float
        list of particle grain diameters in microns
    f_list: list of float
        list of number-density fractions
    s: float
        internal scattering coefficient inside the particle
    constant_D: Boolean
        if False, get_mrp function is used
    include_D: Boolean
        if True, grain size is included in weighted average calculation

    Returns
    -------
    w_mix: 1D numpy float array
        spectrum of the single scattering albedo
    """
    # make sure that all n and k arrays are of the same size as common_wav
    
    for i in range(len(f_list)):
        if n_list[i].size !=  wav_common.size:
            raise ValueError('n_list array at index %d does not have the same dimension as wav_common'%i)
            
    for i in range(len(f_list)):
        if k_list[i].size !=  wav_common.size:
            raise ValueError('k_list array at index %d does not have the same dimension as wav_common'%i)
            
    
    # calculate w for each component
    
    w_list = []
    
    for i in range(len(f_list)):
        w = get_w(n_list[i],k_list[i],wav_common,D_list[i],s, constant_D=constant_D)
        w_list.append(w)
        
    # calculate cross-sectional area for each component
    
    sigma_list = []
   
    for i in range(len(f_list)):
        sigma = D_list[i]**2
        sigma_list.append(sigma)
        
    # calculate w_mix
    
    num = 0.
    den = 0.
    
    for i in range(len(f_list)):
        
        if include_D==True:
            num += f_list[i]*sigma_list[i]*w_list[i]
            den += f_list[i]*sigma_list[i]
        else:
            num += f_list[i]*w_list[i]
            den += f_list[i]
        
    w_mix = num/den
   
    return w_list, w_mix


def get_p_mix(n_list,k_list,wav_common,D_list,f_list,p_type_list,g,s, constant_D):
    """Function to calculate average phase function for a particulate mixture
    of two or more components. 
    
    NOTE: this function requires all n and k arrays to be at
    the resolution of wav_common (use utils.spectra_list_match function)
    
    NOTE: We assume that Q_E=1 for all the components. 

    Parameters
    ----------
    n_list: list of 1D numpy float arrays
        list of spectra of n, the real part of refractive index
    k_list: list of 1D numpy float arrays
        list of spectra of k, the imaginary part of refractive index
    wav_common: 1D numpy float array
        the common wavelength axis for all optical constants arrays
    D_list: float
        list of particle grain diameters in microns
    f_list: list of float
        list of number-density fractions
    p_type_list: list of str
        list of function types for each component. Options are same as the get_p function:
        'Euler', 'HG1' (one parameter Henyey-Greenstein) and 'HG2'
    g: float
        phase angle in degrees
    s: float
        internal scattering coefficient inside the particle
    constant_D: Boolean
        if False, get_mrp function is used 
        
    Returns
    -------
    p_mix: 1D numpy float array
        spectrum of the phase function
    """

    # make sure that all n and k arrays are of the same size as common_wav
    
    for i in range(len(f_list)):
        if n_list[i].size !=  wav_common.size:
            raise ValueError('n_list array at index %d does not have the same dimension as wav_common'%i)
            
    for i in range(len(f_list)):
        if k_list[i].size !=  wav_common.size:
            raise ValueError('k_list array at index %d does not have the same dimension as wav_common'%i)
            
    
    # calculate p for each component
    
    p_list = []
    
    for i in range(len(f_list)):
        
        if p_type_list[i]=='isotropic':
            p = get_p(g,type=p_type_list[i])
        elif p_type_list[i]=='Euler':
            p = get_p(g,type=p_type_list[i])
        elif p_type_list[i]=='HG1':
            raise ValueError('HG1 phase function is currently not supported in the model')
        elif p_type_list[i]=='HG2':
            b = get_b(n_list[i],k_list[i],wav_common,D_list[i],s, constant_D=constant_D)
            c = get_c(n_list[i],k_list[i],wav_common,D_list[i],s, constant_D=constant_D)
            p = get_p(g,type=p_type_list[i],b=b,c=c)
        else:
            raise ValueError('phase function type not recognized')
        p_list.append(p)
        
    # calculate cross-sectional area and w for each component
    
    sigma_list = []
    w_list = []
   
    for i in range(len(f_list)):
        sigma = D_list[i]**2
        sigma_list.append(sigma)
        
        w = get_w(n_list[i],k_list[i],wav_common,D_list[i],s, constant_D=constant_D)
        w_list.append(w)
        
    # calculate p_mix
    
    num = 0.
    den = 0.
    
    for i in range(len(f_list)):
        num += f_list[i]*sigma_list[i]*w_list[i]*p_list[i]
        den += f_list[i]*sigma_list[i]*w_list[i]
        
    p_mix = num/den
   
    return p_list, p_mix