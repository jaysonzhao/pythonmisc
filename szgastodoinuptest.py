# coding=utf8
import threading
import time
import random
import cx_Oracle
import os
import sys


def getUrlA():
    connection = cx_Oracle.connect("PORTAL_TODO", "szgas4321", "10.9.6.66/PTPROD2")
    cursor = connection.cursor()

    insertstatementtemplate = "INSERT INTO PORTAL_TODO.TAB_TODO(SYSCODE, TODOID, TODOPERID, TODOTITLE, TODOCREATEDATE, OPERSTATUS, TODOSTATUS, TODOLEVEL, TODODETAILURL, TODOGENEURL, BAK, \"type\", INSERTTIME, UPDATETIME, ISDEL)\
     VALUES \
      (\
        '031',\
        '00001',\
        :1,\
        'TESTSONGSS',\
        TO_DATE('2018-11-12', 'YYYY-MM-DD'),\
        'ADD',\
        '0',\
        'L',\
        'http://hao123.com',\
        'http://hao123.com',\
        'bb15820e-e16d-491e-9452-31f81b00d600',\
        '0',\
        TO_DATE('2018-11-12', 'YYYY-MM-DD'),\
        TO_DATE('2018-11-12', 'YYYY-MM-DD'),\
        '0'\
      )"

    users = []
    with open('users.txt', 'r') as f:
        for line in f:
            users.append(line.strip('\n'))
    rannum = random.randint(0, len(users) - 1)
    print("Inserted: ", users[rannum])
    cursor.execute(insertstatementtemplate, [users[rannum]])
    connection.commit()
    cursor.close()
    connection.close()


def getUrlB():
    connection = cx_Oracle.connect("PORTAL_TODO", "szgas4321", "10.9.6.66/PTPROD2")
    cursor = connection.cursor()
    operstatus = ["ADD","ADD_R" ]
    todostatus = ["0", "1"]
    updatestatementtemplate = "UPDATE PORTAL_TODO.TAB_TODO\
                                    SET OPERSTATUS = :1,\
                                    TODOSTATUS   = :2 \
                               WHERE \
                                      TODOID = :3\
                                     AND TODOPERID    = :4"
    conds = []
    with open('updatecon.txt', 'r') as f:
        for line in f:
            conds.append(line.strip('\n').split(","))
    
    rannum = random.randint(0, len(conds) - 1)
    ranstat = random.randint(0, 1)
    print("Updated: ", conds[rannum][0])
    cursor.execute(updatestatementtemplate, [operstatus[ranstat], todostatus[ranstat], conds[rannum][0], conds[rannum][1]])
    connection.commit()
    cursor.close()
    connection.close()

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

        if (time.process_time() / 60 > torun):
            runningflag = False
