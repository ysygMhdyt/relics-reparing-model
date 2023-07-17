#ifndef MAINSCENE_H
#define MAINSCENE_H

#include <QMainWindow>
#include "scene.h"

class MainScene : public QMainWindow
{
    Q_OBJECT

public:
    MainScene(QWidget *parent = 0);
    ~MainScene();

    //painter
    void paintEvent(QPaintEvent *event);

    scene *Ascene=NULL;
};

#endif // MAINSCENE_H
