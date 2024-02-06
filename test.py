import numpy
from matplotlib.figure import Figure
from scipy.stats import norm

def test(s, b, sigma, ntoys):
  thetas = numpy.random.lognormal(sigma = sigma , size = (ntoys,))
  thetas[thetas < 0.0] = 0.0
  bs = b * (1.0 + thetas)

  bkgonly = numpy.sort(numpy.random.poisson(bs))
  sigbkg = numpy.random.poisson(s + bkgonly)

  avgsb = numpy.mean(sigbkg)
  idx95 = int(ntoys * 0.95)
  val95 = bkgonly[idx95]
  maxsb = numpy.max(sigbkg)

  fig = Figure((6, 6))
  plt = fig.add_subplot(111)

  plt.hist \
    ( [ bkgonly , sigbkg ]
    , label=["background only", "signal+background"]
    , density=True
    , bins=numpy.mgrid[0:maxsb:maxsb * 1j]
    )
  

  plt.set_xlabel("counts")
  plt.set_ylabel("probability")
  _ , ymax = plt.get_ylim()

  plt.plot([avgsb, avgsb] , [0.0, ymax], alpha=0.75, color="black", label="mean Poiss(s+b)")
  plt.plot([val95, val95] , [0.0, ymax], alpha=0.75, color="red", label="95th quantile Poiss(b)")

  plt.legend()

  fig.savefig("test.png")

  x = numpy.sum(avgsb > bkgonly) / ntoys
  print("Z score:", norm.ppf(1.0-(1.0-x)/2.0))
  return 1.0 - numpy.sum(avgsb > bkgonly) / ntoys


print("expected p-value:\n%0.2f%%" % (test(2, 1, 0.85, 1000000) * 100))
