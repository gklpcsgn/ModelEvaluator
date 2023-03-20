from DETRDetector import *
import time

detector = DETRDetector()

# print time

start_time = time.time()
detector.onImage("/media/gklpcsgn/CE623CD9623CC84B/TYX/man_cafe.jpg")
print("--- %s seconds ---" % (time.time() - start_time))
