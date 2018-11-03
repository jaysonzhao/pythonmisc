import cx_Oracle
import os
import xmltodict
import pandas as pd
import xml.etree.cElementTree as et
from sqlalchemy import types, create_engine




os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8' 

connection = cx_Oracle.connect("csmart", "csmart", "mail.gzsolartech.com/smartformsdb")
cursor = connection.cursor()
cursor.execute("SELECT d.DOCUMENT_ID,d.DOCUMENT_DATA.getClobVal(),d.FORM_NAME,d.APP_ID FROM DAT_DOCUMENT d "
                      "where d.form_name='testwf3'")
result = cursor.fetchone()[1].read() #for oneline test
datajson = xmltodict.parse(result)['root']
dfcols = datajson.keys()
#print(dfcols)
df = pd.DataFrame()

resultall = cursor.fetchall()

for res in resultall:
    etree = et.fromstring(res[1].read())
    #s = pd.Series()
    s = dict()
    for node in etree:
        #s = pd.Series(index=dfcols)
        #print(node.tag, node.text)
        col_name = node.tag
        val = node.text
        if(val==None):
            val = '0'
        s[col_name] = val
        #print(s)
    #print(s)
    dftemp = pd.DataFrame([s], columns=s.keys())
   # df = pd.concat([df, dftemp], axis=0, sort=False).reset_index()
    df = pd.concat([df, dftemp], sort=False)
#print(df)
conn = create_engine('oracle+cx_oracle://csmart:csmart@mail.gzsolartech.com:1521/?service_name=smartformsdb')
dtyp = {c:types.VARCHAR(df[c].str.len().max())
        for c in df.columns[df.dtypes == 'object'].tolist()}
#print(dtyp)
df.to_sql('jaysontest', conn, dtype=dtyp, if_exists='replace', chunksize=50)

