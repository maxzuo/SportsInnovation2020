from model import *
from time import sleep

stream = cv2.VideoCapture('./videos/720_football.mov')

prev_frame = None
averages = None
counter = 0
avg_diff = None

# TEST VIDEO HAS 7 PLAYS
# stream.set(2, 100)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == "__main__":
    deltas = []
    start = datetime.now()
    while True:
        ret, frame = stream.read()
        counter += 1
        # if not averages is None:
        #     if counter > len(averages): break
        #     if averages[counter - 1] < avg_diff * 0.7:
        #         cv2.imshow("play start", frame)
        #         sleep(0.05)
        #     else:
        #         cv2.imshow("play start", np.zeros(frame.shape))
        if not ret and averages is None:
            counter = 0
            averages = moving_average(np.array(deltas[1:]), n=15)
            # averages = np.log(averages)
            averages = np.sqrt(averages)


            avg_diff = np.average(averages)

            plt.plot(range(len(averages)), averages)
            plt.plot([0, len(averages)], [avg_diff, avg_diff], 'g')
            plt.plot([0, len(averages)], [avg_diff*0.8, avg_diff*0.8], 'g')
            plt.show()

            # stream.set(2, 0)
            break
        elif not ret:
            break
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = 128
        gray = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)


        if prev_frame is None:
            prev_frame = gray
        
        delta = gray - prev_frame
        thresh = cv2.threshold(delta, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # cv2.imshow('delta', thresh)
        # cv2.imshow('frame',gray)
        delt_score = np.sum(thresh)
        deltas.append(delt_score)

        prev_frame = gray
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # find best moment
    masked = (averages < avg_diff * 0.7).astype(int)
    # stream = cv2.VideoCapture("videos/720_football.mov")
    last = None
    for i in range(len(masked) - 1):
        # if not last is None and i < last + 30:
        #     continue
        if masked[i] == 1:# and masked[i + 1] == 0:
            print(i)
            # print(stream.get(1))
            stream.set(1, i)
            ret, frame = stream.read()
            if not ret:
                print(frame.shape)
            cv2.imwrite("training_images/%d.png" % i, frame)
            last = i
    
    # print(deltas)
    print(datetime.now() - start)

    


stream.release()
cv2.destroyAllWindows()