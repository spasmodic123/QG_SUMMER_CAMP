from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from flask_migrate import Migrate

app = Flask(__name__)
app.config['DEBUG'] = True
# 在app.comfig设置好连接数据库的信息,然后使用SQLALchemy创建db对象,SQLALchemy会从app.config中读取信息
HOSTNAME = '127.0.0.1'
PORT = 3306
USERNAME = 'root'
PASSWORD = '112233'
DATABASE = 'database_learn'

app.config[
    'SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4'

db = SQLAlchemy(app)
'''# 测试是否连接成功
with app.app_context():
    with db.engine.connect() as conn:
        rs = conn.execute(text("select 1"))
        print(rs.fetchone())  # (1,)'''

migrate = Migrate(app, db)  # 将数据同步到数据库中
# 将数据同步到数据库中的三个步骤
# 1.flask db init 该步骤只需要执行一次
# 2.flask db migrate 识别ORM模型的变化,生成迁移脚本
# 3. flask db upgrade 运行迁移脚本,同步数据
# 一个orm模型代表数据库中的一张表,一个类属性代表表中的字段,一个实例对象代表表中的每条记录


class USER(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), nullable=False)  # String相当于varchar
    password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100),nullable=False)
    signature = db.Column(db.String(100))

# 外键,创建关系型数据库,多个表之间建立联系
class Article(db.Model):
    __tablename__ = 'article'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)

    # 给作者添加外键
    author_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # 因为与之关联的USER表的id也是int,所以要关联integer
    author = db.relationship('USER', backref='articles')
    # 跟其他表建立双向联系,相当于自动在USER表中寻找匹配项,    且backref会自动在USER模型中创建一个articles属性
    # 如此一来,USER对象可以通过articles来访问Article,  Article对象可以通过id访问USER


'''with app.app_context():
    db.create_all()  # 同步到数据库中,这个方法有缺陷,无法同步字段,一般采用migrate'''


@app.route('/')
def hello_world():
    return "hello world"


# add user
@app.route('/user/add')
def add_user():
    # 1.创建ORM对象
    # 2.将对象添加到db.session
    # 3.同步到数据库
    user = USER(username='zhangsan', password='123456')
    db.session.add(user)
    db.session.commit()
    return 'user create success'


# 查询操作
@app.route('/user/query')
def query_user():
    # 1.get方法查找:根据主键查找
    user = USER.query.get(1)
    print(f'{user.id}---{user.username}:{user.password}')
    # 2.使用filter查询,返回Query类列表对象,可以进行切片,循环.访问等操作
    user1 = USER.query.filter_by(username='zhangsan')
    for uesr in user1:
        print(user.username, user.id, user.password)

    # user = USER.query.all()   get all data
    # user = USER.query.first()   get the first data
    return 'query success'


@app.route('/user/edit/')
def update_user():
    user = USER.query.filter_by(username='zhangsan').first()
    # 想要更改对象,不仅要查询到对象,还要获取到对象,所以后面要加first或get
    user.password = '6666666'
    db.session.commit()
    return 'update success'


@app.route('/user/delete/')
def delete_user():
    user = USER.query.get(1)
    db.session.delete(user)
    db.session.commit()
    return 'delete user success'


@app.route('/article/add/')
def add_article():
    article1 = Article(title='HOW_TO_LEARN_FLASK', content='flask_is_about_xxxxxxxxxxx')
    article2 = Article(title='HOW_TO_LEARN_ENGLISH', content='ENGLISH_is_about_xxxxxxxxxxx')
    article1.author = USER.query.get(2)  # 文章对应作者
    article2.author = USER.query.get(2)
    db.session.add_all([article1,article2])
    db.session.commit()
    return 'article add success'


@app.route('/article/query/')
def query_artilce():
    user = USER.query.get(2)  # 因为已经建立了外键,不用直接查找文章,可以通过user.articles查找
    for author in user.articles:
        print(author.title)
    return 'article query success'

app.debug = True
if __name__ == '__main__':
    app.run(debug=True)
