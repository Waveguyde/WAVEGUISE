import numpy as np

def get_basis(x, y, max_order=1):
    #Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc.
    basis = []
    for i in range(max_order+1):
        for j in range(max_order - i +1):
            basis.append(x**j * y**i)
    return basis

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def calculate_2dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

def BG_removal(data, max_order=1):
    
    data-=np.nanmean(data)
    
    ny, nx = data.shape
    y0 = np.arange(ny)#linspace(0, (len(signal[:, 0]) - 1) * 0.625, ny)
    x0 = np.arange(nx)#linspace(0, (len(signal[0, :]) - 1) * 0.625, nx)
    dx, dy = 1, 1#(x0[1] - x0[0]), (y0[1] - y0[0])

    X, Y = np.meshgrid(x0,y0)
    x, y = X.flatten(), Y.flatten()
    b = data.ravel()
    mask = ~np.isnan(b)
    b = b[mask]
    x = x[mask]
    y = y[mask]
    
    basis = get_basis(x, y, max_order)

    A = np.vstack(basis).T
    c, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # Calculate the fitted surface from the coefficients, c.
    fit = np.sum(c[:, None, None] * np.array(get_basis(X, Y, max_order)).reshape(len(basis), *X.shape), axis=0)
    
    detrended_data=data-fit
    detrended_data[np.isnan(detrended_data)]=0

    ft = calculate_2dft(detrended_data)
    freqs_x    = np.fft.fftfreq(nx, 1)
    freqs_x    = np.fft.fftshift(freqs_x)
    freqs_y    = np.fft.fftfreq(ny, 1)
    freqs_y    = np.fft.fftshift(freqs_y)
    freqs_X, freqs_Y = np.meshgrid(freqs_x,freqs_y)

    filtered_ft=ft.copy()
    filtered_ft[int(ny/2)-1:int(ny/2)+2,int(nx/2)-1:int(nx/2)+2]=0
    highpass_data=calculate_2dift(filtered_ft)
    lowpass_data=detrended_data-highpass_data
    
    return highpass_data, fit+lowpass_data 