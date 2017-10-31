import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from bottle import Bottle, ServerAdapter, route, run, request, template
import re
#####################################################################
#####################################################################
lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 2
#hm_data = 2000000

#batch_size = 128
#hm_epochs = 10
#tf.reset_default_graph()
x = tf.placeholder('float')
y = tf.placeholder('float')


current_epoch = tf.Variable(1)

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([27001, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

#####################################################################
#                       DEFINE NETWORK HERE                         #
#####################################################################
def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.tanh(l2)

    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']

    return output
#tf.reset_default_graph()
#saver = tf.train.Saver()
#tf.reset_default_graph()
saver = tf.train.import_meta_graph('./model.ckpt.meta')
#####################################################################
#                       USE NETWORK HERE                            #
#####################################################################
def use_neural_network(input_data):
    prediction = neural_network_model(x)
    with open('lexicon-2500-2638.pickle','rb') as f:
        lexicon = pickle.load(f)
        
    with tf.Session() as sess:
        #tf.reset_default_graph()
        sess.run(tf.global_variables_initializer())
        #tf.reset_default_graph()
        saver.restore(sess,"model.ckpt")
        #print('model restored')
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
                features[index_value] += 1

        features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))

        
        if result[0] == 0:
            print('Positive:',input_data)
            return 'Positive:',prediction.eval(feed_dict={x:[features]})
        elif result[0] == 1:
            #Print('Neg')
            print('Negative:',input_data)
            return 'Negative:',prediction.eval(feed_dict={x:[features]})
            


server_log = 'server.log'


#####################################################################
#                       Server Starts Here                          #
#####################################################################
class MyWSGIRefServer(ServerAdapter):
    server = None

    def run(self, handler):
        from wsgiref.simple_server import make_server, WSGIRequestHandler
        if self.quiet:
            class QuietHandler(WSGIRequestHandler):
                def log_request(*args, **kw): pass
            self.options['handler_class'] = QuietHandler
        self.server = make_server(self.host, self.port, handler, **self.options)
        self.server.serve_forever()

    def stop(self):
        # self.server.server_close() <--- alternative but causes bad fd exception
        self.server.shutdown()
listen_addr='0.0.0.0'

listen_port=10001
app = Bottle()
server = MyWSGIRefServer(host=listen_addr, port=listen_port)


#OLD http://127.0.0.1:12345/posneg?blog=I%20hate%20fish&key=test1

#http://127.0.0.1:10001/posneg?blog=I%20hate%20fish&key=test1
#####################################################################
#                       START APP HERE                              #
#####################################################################


@app.route('/posneg')
def index():
    if request.GET.get('key') == 'test1':
        
        rx = re.compile('\W+')
        res = rx.sub(' ', str(request.GET.get('blog'))).strip()
        try:
            Count = int(open(server_log,'r').read().split('\n')[-2])+1
        except:
            print('first')
            Count=1
        
        
        if Count >= 1000:
            Count = 0
            #print(str('SERVER SHUTDOWN '+ Count))
            print('SERVER SHUTDOWN')
            print(Count)
            with open(server_log,'a') as f:
                f.write(str(Count)+'\n')
            server.stop()
        with open(server_log,'a') as f:
            f.write(str(Count)+'\n')
        print(Count)        
            
            
                        
        return str(use_neural_network(res))
		#str(int(request.GET.get('number1')) + int(request.GET.get('number2')))
    else:
        return 'Unsupported operation'

    
@app.route('/')
def index():
    """Home Page"""
    
    return template("bottleFrontEnd.tpl", message="The answer will appear here.") 


@app.post('/')
def formhandler():
      
    rx = re.compile('\W+')
    res = rx.sub(' ', str(request.forms.get('blog'))).strip()
    try:
        Count = int(open(server_log,'r').read().split('\n')[-2])+1
    except:
        Count=1
           
    if Count >= 1000:
        Count = 0
        #print(str('SERVER SHUTDOWN '+ Count))
        print('SERVER SHUTDOWN')
        print(Count)
        with open(server_log,'a') as f:
            f.write(str(Count)+'\n')
        server.stop()
    with open(server_log,'a') as f:
        f.write(str(Count)+'\n')
    print(Count)        

    
    
    
    wordcount = len(res.split())
    print (wordcount)
    if wordcount<200:
         message =  str('wordcount is below optimum accuracy is reduced! '+str(use_neural_network(res)))
    else:
        message = str(use_neural_network(res))
  
    
    return template("bottleFrontEnd.tpl", message=message)
#####################################################################
#                       APP ENDS HERE                               #
#####################################################################

try:
    app.run(server=server)
except Exception.ex:
    print (ex)

    
#if __name__ == '__main__':
#    run(host='127.0.0.1', port=12345)

#use_neural_network("He's an idiot and a jerk.")
#use_neural_network("This was the best store i've ever seen.")
#use_neural_network("fuck off")
#use_neural_network("I really hate that boy he totally suck")
