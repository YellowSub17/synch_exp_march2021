
import numpy as np

hc = 12398.4  #eV / A
pe = 9000    # photon energy in eV
wl = hc/pe    # wavelength in angstrom

print("wavelength = ", wl, "A")

z = 3.88     # detector sample distance
pw = 75e-6   #pixel width
npix = 70

qmax = 2*np.pi*(2/wl)*np.sin(np.arctan(pw*npix/z)/2.0)
d = 2*np.pi / qmax

print("qmax = ", qmax, " inverse Angstrom")
print("qmax (no 2 pi) = ", qmax/(2*np.pi), " inverse Angstrom")
print("d (resolution) = ", d, "A" )
