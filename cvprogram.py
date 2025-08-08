import cv2

img=cv2.imread("panda.jpeg",0)
gray=cv2.cvtColor(img,cv2.COLOR_BAYER_BG2GRAY)
#cv2.line(gray,(50,50),(300, 300),(0,0 ,0),5)
#cv2.line(gray,(50,50),(50, 300),(698,0 ,0),5)
cv2.rectangle(img, (50, 50), (350, 350), (0, 0, 255), 1)
#cv2.circle(img,(50,50),50,(0,0,255),-1)
cv2.putText(img,'My Pandas',(30,300),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),1)
cv2.imshow("MY Pandas",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#filename="forest.mp4"

#cap=cv2.VideoCapture("forest.mp4")
#while(1):
    #rer,frame=cap.read()
    #cv2.imshow("MY video",frame)
    #k=cv2.waitKey(1)
    #if k==27:
      #  break
#cap.release()
#cv2.destroyAllWindows()
    
    #web cam
#cap=cv2.VideoCapture(0)
#while(1):

#rer,frame=cap.read()
#cv2.imshow("MY video",frame)
#k=cv2.waitKey(1)
#if k==27:
#break
#cap.release()
#cv2.destroyAllWindows()
#sr="http://192.168.195.60:8080/videofeed"
#cap=cv2.VideoCapture(sr)
#while(1):
    #ret,frame=cap.read()
    #cv2.imshow("frame",frame)
   # k=cv2.waitKey(1)
  #  if k==27:
 #       break
#cap.release()
#cv2.destroyAllWindows()