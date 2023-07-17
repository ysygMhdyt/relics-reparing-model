#include "mainscene.h"
#include <QPainter>
#include <QTimer>
//#include <QString>

MainScene::MainScene(QWidget *parent)
    : QMainWindow(parent)
{
    //size
    setFixedSize(1000,600);
    //logo
    setWindowIcon(QIcon(":/scene/logo.jpg"));
    QString qstr = QString::fromLocal8Bit("三星堆文物修复模型");
    setWindowTitle(qstr);
}

//初始场景背景
void MainScene::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    QPixmap pix;
    pix.load(":/scene/main_bg.jpg");
    painter.drawPixmap(0,0,this->width(),this->height(),pix);
    //实例化主场景
    Ascene=new scene;
    //延时进入到主场景
    QTimer::singleShot(3000,this,[=](){
        this->close();
        Ascene->show();
    });
}

MainScene::~MainScene()
{

}
