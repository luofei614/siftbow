import cgi
import json
from search import search
from train import train
import os
import time

def save_uploaded_file (form_field, upload_dir):
    """This saves a file uploaded by an HTML form.
       The form_field is the name of the file input field from the form.
       For example, the following form_field would be "file_1":
           <input name="file_1" type="file">
       The upload_dir is the directory where the file will be written.
       If no file was uploaded or if the field does not exist then
       this does nothing.
    """
    form = cgi.FieldStorage()
    if not form.has_key(form_field): return
    fileitem = form[form_field]
    if not fileitem.file: return
    fout = file (os.path.join(upload_dir, fileitem.filename), 'wb')
    while 1:
        chunk = fileitem.file.read(100000)
        if not chunk: break
        fout.write (chunk)
    fout.close()

def application(env,start_response):
    start_response('200 ok',[('Content-Type','text/html'),('Access-Control-Allow-Origin','*')]);
    lens=env.get('CONTENT_LENGTH','0')
    if lens=="":
        length=0;
    else:
        length=int(lens)
    body=''
    result=[]
    request= cgi.parse_qs(env['QUERY_STRING'])
    action=request.get('action')[0]
    if action=='search':
        #upload image
        if length>0:
    	   fields=cgi.FieldStorage(fp=env['wsgi.input'],environ=env,keep_blank_values=1)
           img = fields['file']
           ext=os.path.basename(img.filename).split('.')[1];
           img_file_path='/tmp/'+str(time.time())+ext
           open(img_file_path, 'wb').write(img.file.read()); 
           result=search(img_file_path)
           #basename
           ret=[]
           for r in result:
               ret.append(os.path.basename(r))
           os.remove(img_file_path)
           return [json.dumps(ret)]
        else:
           return ['error']
    if action=='upload':
    	fields = cgi.FieldStorage(fp=env['wsgi.input'],environ=env,keep_blank_values=1)
        open('./image_data/'+fields['file'].filename, 'wb').write(fields['file'].file.read());
        return ['sucess']
    if action=='train':
        train('./image_data')
        return ['sucess']
    if action=='del':
    	fields=cgi.FieldStorage(fp=env['wsgi.input'],environ=env,keep_blank_values=1)
        os.remove('./image_data/'+fields.getvalue('path'))
        return ['sucess']


