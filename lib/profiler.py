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
    self.startTimes = {}
    self.stopTimes = {}

  def start(self, name):
    print name + " start"
    self.startTimes[name] = time.time()

  def stop(self, name):
    stopTime = time.time()
    print "%s ran - Time Elapsed: %f s" % (name, (stopTime - self.startTimes[name]))
    self.stopTimes[name] = stopTime

  def startOnce(self, name):
    if not self.startTimes.get(name, False):
      self.start(name)

  def stopOnce(self, name):
    if not self.stopTimes.get(name, False):
      self.stop(name)

