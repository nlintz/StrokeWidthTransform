import time

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%2.2f sec' % \
              (te-ts)
        return result

    return timed

class Timer(object):
  def __init__(self):
    self.startTimes = []
    self.stopTimes = []

  def start(self, name=None):
    if name:
      print name + " start"
    self.startTimes.append(time.time())

  def stop(self, name=""):
    stopTime = time.time()
    print "%s ran - Time Elapsed: %f s" % (name, (stopTime - self.startTimes[-1]))
    self.stopTimes.append(stopTime)