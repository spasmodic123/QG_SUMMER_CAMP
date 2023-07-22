from flask import Flask, url_for, request, render_template
from datetime import datetime

app = Flask(__name__)
app.config['DEBUG'] = True


class User():
    def __init__(self, usename, email):
        self.usename = usename
        self.email = email


def date_format(value, format='%Y-%m-%d-%H-%M'):
    return value.strftime(format)


app.add_template_filter(date_format, 'dformat')  # 添加自定义过滤器,参数函数对象,以及名字


@app.route('/')
def hello_world():
    user = User(usename='zhangshan', email='xxxx@xxxx.com')
    person = {
        'name': 'lisi',
        'email': 'lisi@qq.com'
    }
    return render_template('index.html', user=user, person=person)  # 添加对应的文件名,渲染模板,也可以传递对象


@app.route('/blog/<blog_id>')
def blog_detail(blog_id):
    return render_template('blog_detail.html', blog_id=blog_id, username='靓仔111')  # 后端传参


@app.route('/filter/')
def filter():
    user = User(usename='zhangshannnnnn', email='xxxx@xxxx.com')
    now_time = datetime.now()
    return render_template('filter.html', user=user, now_time=now_time)


@app.route('/control')
def control_statement():
    age = 19
    books = [{
        'name': 'sanguoyanyi',
        'author': 'luoguanzhong'},
        {'name': "xiyouji",
         'author': 'wuchengen'
         }]
    return render_template('control.html', age=age, books=books)


# 带有参数的URL
@app.route('/hey!/<username>')  # http://127.0.0.1:5000/hey!/wbxwbx
def hey_hahahahah(username):
    return 'hey handsome boy is %s' % username


@app.route('/project/')  # 结尾带个斜线,如果在网址输入是忘带结尾斜线,系统会自动步骤,定向的规范的URL.  如果结尾不带斜线则没有这个功能
def pro():
    return "我是梁志超他奶奶,梁志超骂我大傻狗!哈哈哈哈哈"


# 查询字符串的方式传参
@app.route('/book/list')
def book_list():
    page = request.args.get(key='page', default=1, type=int)
    return f'you get the page {page}'  # 把当前环境中的变量直接扔进去使用
    # http://127.0.0.1:8000/book/list?page=999


@app.route('/child1/')  # 模板继承
def child1():
    return render_template('child1.html')


@app.route('/child2/')  # 模板继承
def child2():
    return render_template('child2.html')


@app.route('/static')  # 静态文件
def static_demo():
    return render_template('static.html')

# flask 构建URL
with app.test_request_context():
    print(url_for('hello_world'))
    print(url_for('hey_hahahahah', username='mary'))

# host设置为0.0.0.0,让其他电脑可以访问我电脑的flask项目
# port设置端口号
app.debug = True
if __name__ == '__main__':
    app.run(debug=True)  # Running on http://127.0.0.1:5000  127.0.0.1是本地服务器,5000是端口好