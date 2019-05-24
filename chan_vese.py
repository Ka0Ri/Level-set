import skfmm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))

def grad(phi):
    Dx_n = phi - np.roll(phi, 1, axis=1)
    Dx_p = np.roll(phi, -1, axis=1) - phi
    Dy_n = phi - np.roll(phi, 1, axis=0)
    Dy_p = np.roll(phi, -1, axis=0) - phi
    Dx_0, Dy_0 = np.gradient(phi)

    return [Dx_0, Dx_n, Dx_p, Dy_0, Dy_n, Dy_p]
    
def chan_vese_move_curve(init_shape, I, mu, iters=300, dh=1, dt=0.1, dv=1, l1=1, l2=1, v=0, e=1):
    h, w = init_shape.shape
    Hs = []
    phi = -skfmm.distance(init_shape)

    H = 1/2*(1+2/(np.pi)*np.arctan(phi/e))
    Hs.append(H)
    cv2.imwrite(os.getcwd() + "/frames/" + str(0) + ".jpg", 255*H)
    for i in range(1, iters):
        #compute gradient
        Dx_0, Dx_n, Dx_p, Dy_0, Dy_n, Dy_p = grad(phi)
        #compute c1, c2
        c1 = np.sum(H*I)/(np.sum(H))
        c2 = np.sum((1-H)*I)/(np.sum(1-H))
        #update phi
        s = -v - l1*(I-c1)**2 + l2*(I-c2)**2
        Nx = Dx_p/np.sqrt(1e-8 + (Dx_p/dh)**2 + (Dx_0/dh)**2)
        Ny = Dy_p/np.sqrt(1e-8 + (Dy_p/dh)**2 + (Dy_0/dh)**2)
        Dxx_n = mu*(Nx - np.roll(Nx, 1, axis=1))/dh
        Dyy_n = mu*(Ny - np.roll(Ny, 1, axis=0))/dh
        phi = phi + dv*dt*(e/(np.pi*(e**2 + phi**2)))*(Dxx_n + Dyy_n + s)
        H = 1/2*(1+2/(np.pi*np.arctan(phi/e)))
        Hs.append(H)
        cv2.imshow("", H)
        cv2.waitKey(5)
        cv2.imwrite(os.getcwd() + "/frames/" + str(i) + ".jpg", 255*H)
       
        if(np.linalg.norm(Hs[-1] - Hs[-2]) < 0):
            break
    cv2.waitKey(0)
    return Hs

img = cv2.imread(os.getcwd() + "/cropped/1_4279_2066.jpg")
# img = cv2.imread("img.png")
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

hls = cv2.resize(hls, (400, 400))
hue, lig, sat = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
h, w = hue.shape[:]
cv2.imshow("original", 2*hue)
cv2.waitKey(0)
X, Y = np.meshgrid(np.linspace(-1,1, w), np.linspace(-1,1, h))
init_shape = (X+0)**2+(Y+0)**2 - 0.1
# init_shape = np.ones((h, w))
# init_shape[5:h-5, 5:w-5] = - 1
phies = chan_vese_move_curve(init_shape, 2*hue, dv=1000, mu=1)
# for i in range(len(phies)):
#     plt.contour(X.T, Y.T, phies[i].T, [0], linewidths=(1), colors='black')
#     # plt.show()
# plt.contour(X.T, Y.T, phies[-1].T, [0], linewidths=(1), colors='red')
# plt.show()


