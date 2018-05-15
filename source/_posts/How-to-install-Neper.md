---
title: Neper的安装笔记
date: 2018-05-15 22:51:00
tags:
---

* Neper安装的难点在于gmsh的安装。
----

1. 安装g++,gfortran,cmake
2. 安装GSL,NLopt,libScotch,pthread,POV-Ray。这些可以从Ubuntu应用中心中找找看有没有，如果有的
话就在应用中心安装。这样比较方便还不容易出错。
3. 下载Gmsh源代码包，编译安装。根据Gmsh包内的安装说明安装.编译时会提示没有一些依赖文件，这时候可以挨个
的安装这些依赖文件。直到最后完全编译通过。使用cmake-gui进行文件配置.
4. 最后一步为文件的关联，sudo ln /home/usr/***/编译产生的gmsh文件/bin/gmsh /usr/bin/gmsh