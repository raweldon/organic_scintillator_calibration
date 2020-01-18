
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from scipy.interpolate import interp1d
import lmfit
import os.path
import time

start_time = time.time()
    
def get_meas_data(det_no, f_in, save_dir):   
    if det_no == 0:
        data = np.load(save_dir + f_in, allow_pickle=True, encoding='latin1')[0]
    else:
        data = np.load(save_dir + f_in, allow_pickle=True, encoding='latin1')[1]

    max_data = 80000    
    bin_width = max_data/500.
    meas_hist, meas_bin_edges = np.histogram(data, bins=np.arange(0, max_data + bin_width, bin_width))
    meas_bin_centers = (meas_bin_edges[:-1] + meas_bin_edges[1:])/2 
 
    meas_data = np.array((meas_bin_centers, meas_hist))

    #plt.figure()
    #plt.plot(meas_bin_centers, meas_hist)
    #plt.show()
    return meas_data    

def fit_range(low_bound, high_bound, data_in, name):
    '''select data for fit'''
    print( '\nbounds:',name,low_bound,high_bound)
    fit_range = (low_bound, high_bound)
    low_index = np.searchsorted(data_in[0], fit_range[0])
    high_index = np.searchsorted(data_in[0], fit_range[1])
    data_out = np.array((np.zeros(high_index-low_index), np.zeros(high_index-low_index)))
    data_out[0], data_out[1] = data_in[0][low_index:high_index], data_in[1][low_index:high_index]
    return data_out

def gaussian(x, mu, sigma, A):
    return A/np.sqrt(2.*np.pi*sigma**2) * np.exp(-(x - mu)**2/(2.*sigma**2))

def gaussian_smear(x, y, alpha, beta, gamma, c1, c2):
    '''smears bin centers (x) and bin contents (y) by a gaussian'''
    y_update = np.zeros(len(y))
    for i in range(len(x)):
        x_val = x[i]
        smear_val = np.sqrt((alpha*x_val)**2. + beta**2.*x_val + gamma**2.) 
        gaus_vals = gaussian(x, x_val, smear_val, 1)
        y_update[i] = np.dot(y, gaus_vals)
    return y_update
    
def shift_and_scale(args, bin_data, lin_scaling):
    '''
    shifts and scales the guassian
    bin_data is an (x,y) numpy array of bin centers and values
    shift allows for a horizontal shift in the bin centers (e.g. bin centers (0.5, 1, 1.5) --> (1, 1.5, 2) )
    spread allows for horizontal scaling in the bin centers (e.g. bin centers (0.5, 1, 1.5) --> (1, 2, 3) )
    '''
    alpha, beta, gamma, shift, spread, c1, c2, y_scale = args
    x, y = bin_data
    y = gaussian_smear(x, y, alpha, beta, gamma, c1, c2)
    if lin_scaling:
        y *= c1
    else:
        y *= c1*x**c2
    x = copy.deepcopy(x)
    x = x*spread + np.ones(len(x))*shift
    return x, y

def lin_scaling(shift, spread, bin_data):
    ''' scales uncalibrated measured data '''
    x, y = bin_data
    x = copy.deepcopy(x)
    x = (x - np.ones(len(x))*shift)/spread
    return x, y
    
def calc_chisq(sim_x, sim_y, data_x, data_y):
    # bin centers in simulation are different from measurement -> use linear interpolation to compute SSE
    interp = interp1d(sim_x, sim_y, bounds_error=False, fill_value=0)
    sim_vals = interp(data_x)
    res = data_y - sim_vals
    chisq = [x**2/sim_vals[i] for i, x in enumerate(res)]  
    return chisq

def minimize(fit_params, *args):
    '''
    Minimization based on length of input arguements 
    sim_data is an (x, y) of the simulated bin centers and bin contents
    meas_data is an (x, y) of the measured bin centers and contents
    '''
    pars = fit_params.valuesdict()
    alpha_1 = pars['alpha_1']
    beta_1 = pars['beta_1']
    gamma_1 = pars['gamma_1']
    c1_1 = pars['c1_1']
    c2_1 = pars['c2_1']
    shift = pars['shift']
    spread = pars['spread']

    y_scale_1 = pars['y_scale_1']
    sim_data, meas_data = args[0]
    sim_x_1, sim_y_1 = shift_and_scale((alpha_1, beta_1, gamma_1, shift, spread, c1_1, c2_1, y_scale_1), sim_data, args[1])
    data_x_1, data_y_1 = meas_data
    chisq_1 = calc_chisq(sim_x_1, sim_y_1, data_x_1, data_y_1)     

    return chisq_1
  
def spectra_fit(fit_params, *args, **kwargs):
    print( '\nperforming minimization')

    fit_kws={'nan_policy': 'omit'}
    sim_data_1, meas_data_full_1, meas_data_1 = args[0]
    lin_scaling = args[1]
    print( '    single sprectrum fit')
    res = lmfit.minimize(minimize, fit_params, method='leastsq', args=((sim_data_1, meas_data_1), lin_scaling), **fit_kws)

    if kwargs['print_info']:    
        print( '\n',res.message)
        print( lmfit.fit_report(res) )
 
    if kwargs['show_plots']:
        plot_fitted_spectra( res.params['shift'].value, res.params['spread'].value, lin_scaling, 
                                ( (res.params['alpha_1'].value,), (res.params['beta_1'].value,), (res.params['gamma_1'].value,),
                                (res.params['c1_1'].value,), (res.params['c2_1'].value,),
                                (res.params['y_scale_1'].value,), (sim_data_1,), (meas_data_full_1,), (meas_data_1,) ) )

    # get shift term
    shift_term = res.params['shift'].value
    spread_term = res.params['spread'].value
    
    return shift_term, spread_term

def plot_fitted_spectra(shift, spread, lin_scale, args):
    for index, (alpha, beta, gamma, c1, c2, y_scale, sim_data, meas_data_full, meas_data) in enumerate(zip(args[0], args[1], 
                                                                                                           args[2], args[3], 
                                                                                                           args[4], args[5],
                                                                                                           args[6], args[7], args[8])):
        # update sim data
        sim_new = shift_and_scale((alpha, beta, gamma, shift, spread, c1, c2, y_scale), sim_data, lin_scale)
       
        # plot measured and fitted simulated data
        plt.figure()
        plt.plot(meas_data_full[0], meas_data_full[1], linestyle='None', marker='x', markersize=5, alpha=0.8, label='measured')
        plt.plot(sim_new[0], sim_new[1], '--', label='fit')
        plt.xlabel('ADC units', fontsize=20)
        plt.ylabel('Counts', fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(9000, 14500)
        plt.ylim(0, 15000)
        plt.legend(fontsize=18) 
        plt.tight_layout()
         
        # update measured data and plot
        meas_scaled = lin_scaling(shift, spread, meas_data_full)
        sim_with_res = shift_and_scale((alpha, beta, gamma, 0, 1, c1, c2, y_scale), sim_data, lin_scale)

        plt.figure()            
        plt.plot(meas_scaled[0], meas_scaled[1], linestyle='None', marker='x', markersize=5, alpha=0.5, label='measured')
        plt.plot(sim_with_res[0], sim_with_res[1], '--', label='sim with res')
        plt.plot(sim_data[0], sim_data[1], alpha=0.3, label='sim data')      
 
        # plot edge locations
        if index == 0:
            a = 0.3
            y_min, y_max = plt.ylim()
            y_range = np.arange(0, 5000 + 100, 100.)
            plt.plot([0.478]*len(y_range), y_range, 'k--', alpha=a)
            plt.text(0.478, y_max - y_max/15, 'cs edge')

        plt.ylim(1, 80000)
        plt.xlim(0, 0.7)
        plt.xlabel('ql (MeVee)')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.legend()

def main(fin, det_no, spread, min_range, lin_scaling):
    '''
    Use for individual spectrum fits to get rough idea of initial values and limits for simultaneous fitting
    Note: shift term is only accurate for simulataneous fit, do not use here
    '''

    iso = 'cs'
    cwd = os.getcwd()

    # load sim data
    if det_no == 0:
        ranges = [min_range, 14500] # stilbene
        sim_data = np.load(cwd + '/stilbene_simulated_spectrum.npy')
        print( '\nloading stilbene stimulated spectrum')
    if det_no == 1:
        ranges = [7000, 22000]
        sim_data = np.load(cwd + '/ej309_simulated_spectrum.npy')
        print( '\nloading ej309 simulated spectrum')
    sim_data = [sim_data[0][20:], sim_data[1][20:]]
    print( 'sim data loaded')

    meas_data_full = get_meas_data(det_no, fin, cwd)
    print( 'meas data loaded')
    meas_data = fit_range(ranges[0], ranges[1], meas_data_full, iso)  #0.35,0.7
    data = [sim_data, meas_data_full, meas_data]            

    # lmfit (curve fit wrapper, default: leastsq Levenberg-Marquardt)
    # Only fit beta (Kornilov does this)
    fit_params = lmfit.Parameters()
    fit_params.add('alpha_1', value=0.0, min=0., max=20., vary=False)
    fit_params.add('beta_1', value=0.04, min=0.0035, max=0.1, vary=True)
    fit_params.add('gamma_1', value=0.0, min=0., max=20, vary=False)
    fit_params.add('shift', value=-1000, vary=True)
    fit_params.add('spread', value=spread, min=0, max=50000, vary=True) # 4mev 180000  
    if lin_scaling:
        fit_params.add('c1_1', value=0.01, vary=True)
        fit_params.add('c2_1', value=0., vary=False)
    else:     
        fit_params.add('c1_1', value=0.01, vary=True)
        fit_params.add('c2_1', value=-1.0, vary=True)
    fit_params.add('y_scale_1', value=0.01)
        
    e0, c = spectra_fit(fit_params, data, lin_scaling, print_info=True, show_plots=True)
    return e0, c


if __name__ == '__main__':
    ''' 
    Set lin_scaling to true for linear amplitude scaling (EJ-309), 
        False for power law amplitude scaling (stilbene)
    '''
    fin = '/cs_hists_coinc_t2.npy'
    det_no = 0  # 0 stilbene, 1 ej309
    spread = 28000 # initial guess for spread
    min_range = 9000
    e0, c = main(fin, det_no, spread, min_range, lin_scaling=False)

    # print 477 keV estimate
    val = c*(0.477 + e0/c)
    print( '\n477 keV ADC value =', round(val, 3))
    
    print( "--- %s seconds ---" % (time.time() - start_time))
    plt.show()
