策略加密

#windows 下加密并运行

1.安装Visual StudioComunity 2017，下载地址：

    https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/
    安装时请勾选“使用C++的桌面开发”。
 
2. 在Python环境中安装Cython，打开cmd后输入运行pip install cython即可。

3. 在”管理员”模式的命令行窗口，在策略所在目录，运行：

    cythonize -i demo_strategy.py
    
    编译完成后，Demo文件夹下会多出2个新的文件，其中就有已加密的策略文件demo_strategy.cp37-win_amd64.pyd 
    
    改名=> demo_strategy.pyd
    
    放置 demo_strategy.pyd到windows 生产环境的 strateies目录下。

#centos/ubuntu 下加密并运行


1. 在Python环境中安装Cython，运行pip install cython即可。

3. 在策略所在目录，运行：

    cythonize -i demo_strategy.py
    
    编译完成后，Demo文件夹下会多出2个新的文件，其中就有已加密的策略文件demo_strategy.cp37-win_amd64.so
    
    改名=> demo_strategy.so
    
    放置 demo_strategy.so 到centos/ubuntu 生产环境的 strateies目录下。
