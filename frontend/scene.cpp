#include <QPainter>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QFile>
#include <QFileDialog>
#include <QMessageBox>
#include <QProcess>
#include <QDesktopServices>
#include <QMenu>
#include <QMenuBar>

#include "scene.h"
#include "connectus.h"
#include "introduction.h"
#include <Python.h>
#include <iostream>

using namespace std;

// 修复图片函数
std::string repair(std::string path)
{
    Py_Initialize();
    // 检查初始化是否成功
    if (!Py_IsInitialized())
    {
        cout << "初始化失败" << endl;
        Py_Finalize();
    }

    PyObject *pModule;      //python file
    PyObject*pFunc = NULL;  //python function
    PyObject*pArg = NULL;   //python return
    PyRun_SimpleString("import sys");
    //设置python模块，搜寻位置，文件放在.cpp文件一起
    PyRun_SimpleString("sys.path.append('./')");

    //需调用的Python文件名
    pModule = PyImport_ImportModule("forfront");
    if (!pModule)
    {
        cout << "py文件导入失败" << endl;
        Py_Finalize();
    }
    else
    {
        //调用python文件中的函数
        pFunc = PyObject_GetAttrString(pModule, "repair_img");
        if (!pFunc)
        {
            cout << "函数导入失败" << endl;
            Py_Finalize();
        }
        //c++类型转python类型
        PyObject* pyParams = Py_BuildValue("(s)",path.c_str());

        char * result1;
        //调用函数
        pArg = PyEval_CallObject(pFunc, pyParams);
        //python类型转c++类型
        PyArg_Parse(pArg, "s", &result1);

        cout << result1<< endl;
        return result1;
    }
}

// 将QString转换为std::string
void QString2stdString(QString qs, std::string &str)
{
    str = qs.toLocal8Bit().constData();
}

// 获取文件目录
QString getTheDirectory(QString path)
{
    int i = path.lastIndexOf("/");
    QString fileDirectory = path.left(i);

    return fileDirectory;
}

// 获取文件名
QString getTheFileName(QString path)
{
    int i = path.lastIndexOf("/");
    QString fileName = path.right(path.length()-i-1);

    return fileName;
}

scene::scene(QWidget *parent) : QMainWindow(parent)
{
    // size
    this->setFixedSize(1000,600);
    // logo
    this->setWindowIcon(QIcon(":/scene/logo.jpg"));

    // 所有汉字必须添加QString::fromLocal8Bit，否则会显示乱码
    QString qstr = QString::fromLocal8Bit("三星堆文物修复模型");
    this->setWindowTitle(qstr);

    // 介绍栏与联系方式实例
    Introduction *introd=new Introduction;
    ConnectUs *con=new ConnectUs;
    // 菜单栏
    QMenuBar *menu=menuBar();
    QMenu *start=menu->addMenu(QString::fromLocal8Bit("开始"));
    QAction *repairbtn=start->addAction(QString::fromLocal8Bit("修复"));
    connect(repairbtn,&QAction::triggered,[=]()
    {

    }
    );
    QMenu *Help=menu->addMenu(QString::fromLocal8Bit("帮助"));
    QAction *intro=Help->addAction(QString::fromLocal8Bit("使用说明"));
    connect(intro,&QAction::triggered,[=]()
    {
        introd->show();
    }
    );
    // 分割线
    Help->addSeparator();
    // 联系我们
    QAction *Connection=Help->addAction(QString::fromLocal8Bit("联系我们"));
    connect(Connection,&QAction::triggered,[=]()
    {
        con->show();
    }
    );

    QLineEdit *l=new QLineEdit(this);
    l->setFixedWidth(640);
    l->move(120,70);
    // 设置清除按钮
    l->setClearButtonEnabled(true);
    // 加条提示语句
    QLabel *tipLab1=new QLabel(this);
    tipLab1->setText(QString::fromLocal8Bit("请选择要修复的图片:"));
    tipLab1->setGeometry(120,40,200,30);

    QPushButton *btn=new QPushButton(this);
    btn->setText(QString::fromLocal8Bit("上传图片"));
    btn->move(790,68);
    btn->setStyleSheet("QPushButton{font:20px}");
    btn->resize(100,32);

    QLabel *lab1=new QLabel(this);
    QLabel *lab2=new QLabel(this);

    lab1->setGeometry(120,170,330,330);
    lab2->setGeometry(570,170,330,330);
    lab1->setStyleSheet("QLabel{border:2px dashed #242424;}");
    lab2->setStyleSheet("QLabel{border:2px dashed #242424;}");

    QLabel *tipLab2=new QLabel(this);
    tipLab2->setText(QString::fromLocal8Bit("待修复图片:"));
    tipLab2->setGeometry(120,140,200,30);   // 若放在正中则是250
    QLabel *tipLab3=new QLabel(this);
    tipLab3->setText(QString::fromLocal8Bit("修复后图片:"));
    tipLab3->setGeometry(570,140,200,30);   // 若在正中则是700

    // 待修复图片与修复后图片
    QImage *img1 = new QImage;
    QImage *img2 = new QImage;

    //点击上传文件
    connect(btn,&QPushButton::clicked,[=](){
        QString path=QFileDialog::getOpenFileName(this,QString::fromLocal8Bit("打开文件"),"C:\\");
        l->setText(path);

        //上传图片并对齐label
        img1->load(path);
        img1->scaled(lab1->size(),Qt::KeepAspectRatio);
        lab1->setScaledContents(true);
        lab1->setPixmap(QPixmap::fromImage(*img1));

        string str = path.toStdString();
        // 修改类私有成员
        img_path.assign(str);
    });

    //点击修复图片
    QPushButton *repairBtn = new QPushButton(this);
    repairBtn->setText(QString::fromLocal8Bit("点击修复"));
    repairBtn->move(460,335);
    repairBtn->setStyleSheet("QPushButton{font:20px}");
    repairBtn->resize(100,32);

    connect(repairBtn,&QPushButton::clicked,[=](){
        // 未选中待修复图片时
        if(img_path.empty())
        {
            QMessageBox msg(this);
            msg.setWindowTitle(QString::fromLocal8Bit("错误提示"));
            msg.setText(QString::fromLocal8Bit("尚未选中待修复图片!"));
            msg.setIcon(QMessageBox::Critical);
            msg.setStandardButtons(QMessageBox::Ok);
            if(msg.exec()==QMessageBox::Ok)
            {
                cout<<"Ok is clicked!"<<endl;
            }
        }
        else
        {   // 选中待修复图片后
            string newpath_std = repair(img_path);
            QString newpath = QString::fromStdString(newpath_std);

            img2->load(newpath);
            img2->scaled(lab2->size(),Qt::KeepAspectRatio);
            lab2->setScaledContents(true);
            lab2->setPixmap(QPixmap::fromImage(*img2));
            // 提示框
            QMessageBox msg(this);
            msg.setWindowTitle(QString::fromLocal8Bit("结果提示"));
            msg.setText(QString::fromLocal8Bit("图片修复成功，且已保存至原目录!是否立刻打开图片？"));
            msg.setIcon(QMessageBox::Information);
            msg.setStandardButtons(QMessageBox::Yes|QMessageBox::No);
            if(msg.exec()==QMessageBox::Yes)
            {
                cout<<"Yes is clicked!"<<endl;
                QDesktopServices::openUrl(QUrl(newpath,QUrl::TolerantMode));
            }
            else
            {
                cout<<"No is clicked!"<<endl;
            }
        }
    });

    // 打开修复后文件目录
    QLabel *tipLab4=new QLabel(this);
    tipLab4->setText(QString::fromLocal8Bit("点击此按钮打开修复后图片所在目录:"));
    tipLab4->setGeometry(540,520,260,30);

    QPushButton *openDirectortyBtn = new QPushButton(this);
    openDirectortyBtn->setText(QString::fromLocal8Bit("打开目录"));
    openDirectortyBtn->move(800,520);
    openDirectortyBtn->setStyleSheet("QPushButton{font:20px}");
    openDirectortyBtn->resize(100,32);

    connect(openDirectortyBtn,&QPushButton::clicked,[=](){
        if(img_path.empty())
        {
            QMessageBox msg(this);
            msg.setWindowTitle(QString::fromLocal8Bit("错误提示"));
            msg.setText(QString::fromLocal8Bit("尚无待修复图片!"));
            msg.setIcon(QMessageBox::Critical);
            msg.setStandardButtons(QMessageBox::Ok);
            if(msg.exec()==QMessageBox::Ok)
            {
                cout<<"Ok is clicked!"<<endl;
            }
        }
        else
        {
            // 获取路径
            QString path = QString::fromStdString(img_path);
            QString directory = getTheDirectory(path);
            QDesktopServices::openUrl(QUrl(directory,QUrl::TolerantMode));
        }
    });
}

// 显示背景
void scene::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    QPixmap pix;
    pix.load(":/scene/bg.jpeg");
    painter.drawPixmap(0,0,this->width(),this->height(),pix);
}

