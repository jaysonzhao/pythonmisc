import cx_Oracle
import os
import xmltodict
import pandas as pd
import xml.etree.cElementTree as et
import json
from sqlalchemy import types, create_engine
import time
import ast

time_start=time.time()
connection = cx_Oracle.connect("csmart", "csmart", "mail.gzsolartech.com/smartformsdb")
cursor = connection.cursor()
cursor.execute("select trow.row_data.getClobVal() as rowdata,trow.row_id,trow.table_id, "
               "trow.document_id,trow.sort_num,trow.data_fmt, trow.table_version,doc.form_name,"
               "to_char(trow.row_json_Data )   from dat_table_row trow, dat_document doc where "
               "trow.document_id= doc.document_id  and trow.data_fmt is not null "
               "and (trow.row_data.getClobVal()  is not null or "
               "(trow.row_json_Data is not null and  to_char(trow.row_json_Data) !='None'))")
               #" and doc.form_name='sjltestform' and  rownum  <2   and trow.data_fmt != 'xml' ")

arr=['row_id','table_id','sort_num','document_id','table_version']
df = pd.DataFrame()
df1 = pd.DataFrame()
resultall = cursor.fetchall()

for res in resultall:

    s = dict()
    s1 = dict()
    if res[5] == 'xml':

        etree = et.fromstring(res[0].read());

        for n in etree.iter('row'):

            for child in n:
                val = child.text
                col_name = child.tag
                l=0

                for child1 in child:
                    val = child1.text
                    l=1
                    s1['id'] = res[1]
                    s1['vulse'] = child1.text
                    s1['name'] = col_name
                    dftemp1 = pd.DataFrame([s1], columns=s1.keys())
                    df1 = pd.concat([df1, dftemp1], sort=False)


                #col_name = child.tag
                #s[col_name] = val
                s['id'] = res[1]
                s['name'] = child.tag
                s['form_name'] = res[7]
                s['document_id'] = res[3]
                if l != 1:

                    s1['id'] = res[1]
                    s1['vulse'] = val
                    s1['name'] = col_name
                    dftemp1 = pd.DataFrame([s1], columns=s1.keys())
                    df1 = pd.concat([df1, dftemp1], sort=False)


            dftemp = pd.DataFrame([s], columns=s.keys())
            df = pd.concat([df, dftemp], sort=False)

    else:

        #print res[1]
        #print res[8]
        sValue = json.loads(res[8])
        for k in sValue.keys():

            s['id'] = res[1]
            s['name'] = k
            s['form_name'] = res[7]
            s['document_id'] = res[3]
            dftemp = pd.DataFrame([s], columns=s.keys())
            df = pd.concat([df, dftemp], sort=False)

            a=isinstance(sValue[k], list)
            if(a):
                jsontemp= ast.literal_eval(str(sValue[k]))

                for item in jsontemp:

                    s1['id'] = res[1]
                    s1['vulse'] = str(item)
                    s1['name'] = k
                    dftemp1 = pd.DataFrame([s1], columns=s1.keys())
                    df1 = pd.concat([df1, dftemp1], sort=False)


            if a!=True:

                s1['id'] = res[1]
                item=sValue[k];
                s1['vulse'] = str(item)
                s1['name'] = k
                dftemp1 = pd.DataFrame([s1], columns=s1.keys())
                df1 = pd.concat([df1, dftemp1], sort=False)


    #for i in range(len(arr)):

     #   s['id'] = res[1]
     #   s['name'] = arr[i]
     #   s['form_name'] = res[7]
     #   s['document_id'] = res[3]
     #   dftemp = pd.DataFrame([s], columns=s.keys())
     #   df = pd.concat([df, dftemp], sort=False)

     #   s1['id'] = res[1]
     #   s1['name'] = arr[i]

     #   item = res[i+2];
     #   s1['vulse'] = str(item)
     #   dftemp1 = pd.DataFrame([s1], columns=s1.keys())
     #   df1 = pd.concat([df1, dftemp1], sort=False)



print df
print df1
#conn = create_engine('oracle+cx_oracle://csmart:csmart@mail.gzsolartech.com:1521/?service_name=smartformsdb')


#dtyp = {c:types.VARCHAR(df1[c].str.len().max())
#        for c in df1.columns[df1.dtypes == 'object'].tolist()}

#df1.to_sql('rowstablevaule', conn, dtype=dtyp, if_exists='replace', chunksize=100)


#dtyp = {c:types.VARCHAR(df[c].str.len().max())
      #  for c in df.columns[df.dtypes == 'object'].tolist()}
#df.to_sql('rowstablekey', conn, dtype=dtyp, if_exists='replace', chunksize=50)



time_end=time.time()
print('time cost: ',time_end-time_start, 's')