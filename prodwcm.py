#coding=utf8
import requests
import xmltodict
import cx_Oracle
import os  
print('starting')
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8' 

connection = cx_Oracle.connect("PORTAL", "szgas4321", "10.9.6.56/ptprod")
cursor = connection.cursor()
#cursor.execute("SELECT contentname,contenttitle FROM TAB_NEWS  WHERE siteareaname='shen3ji4tian1di4' ORDER BY approvetime DESC")
cursor.execute("SELECT contentname,contenttitle FROM TAB_NEWS order by approveTime desc")

counting = 1
deleted = 1    
for contentname, contenttitle in cursor:
    try:
       #get content item feed
       r = requests.get("http://10.9.6.43:10059/wps/mycontenthandler/wcmrest/query?title="+contenttitle,auth=('super','admin123'))
       print('total: '+str(counting))
       counting+=1
       queryres = xmltodict.parse(r.text)
       feed = queryres['feed']
       entry = feed['entry']
       #print entry and len to compare the diff for content return by wcm
       #print('entrylenth: '+str(len(entry)))
       #print(entry)
       for item in entry:
           wcmitemid = item['id']
           wcmitemid = wcmitemid[8:]
           wcmitemname = item['wcm:name']
           wcmtitle = item['title']
           print('processing id:'+wcmitemid+'  name:'+wcmitemname+' title:')
           print(wcmtitle)
           checkdup = connection.cursor()
           checkdup.execute("SELECT contentname FROM TAB_NEWS where contentname=:wcmnametodel",wcmnametodel=wcmitemname)
           checkdup.fetchall()
           if checkdup.rowcount == 0:
                print("deleting:", wcmitemname, wcmitemid)
                r = requests.delete('http://10.9.6.45:10059/wps/mycontenthandler/wcmrest/Content/'+wcmitemid,auth=('super','admin123'))
                if r.status_code == 200:
                    print('deleted: '+str(deleted))
                    deleted+=1
                else:
                    print('delete opr error')
           else:
               print(contenttitle+' duplicated.')
   
    except:
        print('skipping '+ contentname)
        continue

connection.close()
    

