import cgi
import json
from search import search
def application(env,start_response):
    start_response('200 ok',[('Content-Type','text/html'),('Access-Control-Allow-Origin','*')]);
    lens=env.get('CONTENT_LENGTH','0')
    if lens=="":
        length=0;
    else:
        length=int(lens)
    body=''
    result=[]
    fields=cgi.FieldStorage(fp=env['wsgi.input'],environ=env,keep_blank_values=1)
    if 'search' in fields:
        result=search(fields.getvalue('search'))
    return [json.dumps(result)]
