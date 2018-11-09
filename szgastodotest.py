import requests
import threading
import time


def getUrlA():
    url = 'http://10.9.6.44:10039/wps/portal/!ut/p/z1/04_Sj9CPykssy0xPLMnMz0vMAfIjo8ziDVCAo4FTkJGTsYGBu7OJfjgWBchK9aNI14-iIAq_8V54LbAwAPnAqMjX2TddP6ogsSRDNzMvLV8_Iic_Pb-0BGh7FCH9UYRcEOiTn56emuIPNK4gNzSiyictOCBdUREA1Ejwcg!!/dz/d5/L2dBISEvZ0FBIS9nQSEh/p0/IZ7_00000000000000A0BR2B300I81=CZ6_00000000000000A0BR2B300GC4=LA0=Eaction!wps.portlets.login==/#Z7_00000000000000A0BR2B300I81'
    d = {'wps.portlets.userid': 'SONGSS', 'password': 'Songsisi1989'}
    r = requests.post(url, data=d, headers={'Content-Type': 'application/x-www-form-urlencoded'})
    r1 = requests.get('http://10.9.6.44:10039/wps/myportal/Todo/todo/findAll2.action?perPage=500&currentPage=1')
    print(r1.text)


def getUrlB():
    url = 'http://10.9.6.44:10039/wps/portal/!ut/p/z1/04_Sj9CPykssy0xPLMnMz0vMAfIjo8ziDVCAo4FTkJGTsYGBu7OJfjgWBchK9aNI14-iIAq_8V54LbAwAPnAqMjX2TddP6ogsSRDNzMvLV8_Iic_Pb-0BGh7FCH9UYRcEOiTn56emuIPNK4gNzSiyictOCBdUREA1Ejwcg!!/dz/d5/L2dBISEvZ0FBIS9nQSEh/p0/IZ7_00000000000000A0BR2B300I81=CZ6_00000000000000A0BR2B300GC4=LA0=Eaction!wps.portlets.login==/#Z7_00000000000000A0BR2B300I81'
    d = {'wps.portlets.userid': 'SONGSS', 'password': 'Songsisi1989'}
    r = requests.post(url, data=d, headers={'Content-Type': 'application/x-www-form-urlencoded'})
    r2 = requests.get('http://10.9.6.44:10039/wps/myportal/Todo/todo/findAllSum.action')
    print(r2.text)


if __name__ == '__main__':

    a = input("Max Thread Number:")
    torun = float(input("Run Period in Minutes:"))
    runningflag = True
    # starttime = time.clock()
    while (runningflag):
        thread_listA = []
        thread_listB = []
        for i in range(int(a)):
            tA = threading.Thread(target=getUrlA)
            tB = threading.Thread(target=getUrlB)
            thread_listA.append(tA)
            thread_listB.append(tB)

        for tA, tB in zip(thread_listA, thread_listB):
            tA.setDaemon(True)
            tB.setDaemon(True)
            tA.start()
            tB.start()

        for tA, tB in zip(thread_listA, thread_listB):
            tA.join()
            tB.join()
        # print(time.process_time())
        #endtime = time.clock()
        #runtime = (endtime-starttime)/60
        if(time.process_time()/60>torun):
            runningflag = False