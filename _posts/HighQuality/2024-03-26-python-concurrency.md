---
title: python并发编程
subtitle: 
date: 2024-04-28 21:01:37 +0800
categories:
  - good_artical
tags:
  - python
  - 代码解析
published: true
image:
---
* content
{:toc}


python 多线程因为存在GIL锁，而基本上没有用，基本上只能用于较多 IO 的情况，然后CPU只运行一个线程。


常用的方式是使用多进程：
# 多进程
## 多进程模块

主要是multiprocessing包，来实现（因为太依赖操作系统底层了）。操作的方式和`threading.Thread` 差不多。不同的是，多线程可以共享进程内的资源和对象。而多进程需要通过进程间通信的方式传递 **共同处理的资源和对象**。

## 多进程实现方法

实现方法和多线程类似，使用`multiprocessing.Process`创建*进程对象*，然后这些进程对象包含`start()`, `run()`, `join()`方法，其中有一个方法不同Thread线程对象中的守护线程方法是setDeamon，而Process进程对象的守护进程是通过设置daemon属性来完成的。

主要包含两种方式。
- 通过`Process(fun, args)`初始化，
- 通过集成`Process`类然后重写`run()`方法

示例
```python
from multiprocessing import  Process

# 方法1
def fun1(name):
    print('测试%s多进程' %name)

p = Process(target=fun1,args=('Python',)) #实例化进程对象
p.start()


# 方法2
class MyProcess(Process): #继承Process类
    def __init__(self,name):
        super(MyProcess,self).__init__()
        self.name = name

    def run(self):
        print('测试%s多进程' % self.name)

p = MyProcess('Python') #实例化进程对象
p.start()
```


## Process类介绍
```bash
构造方法：

Process([group [, target [, name [, args [, kwargs]]]]])
　　group: 线程组 
　　target: 要执行的方法
　　name: 进程名
　　args/kwargs: 要传入方法的参数

实例方法：
　　is_alive()：返回进程是否在运行,bool类型。
　　join([timeout])：阻塞当前上下文环境的进程程，直到调用此方法的进程终止或到达指定的timeout（可选参数）。
　　start()：进程准备就绪，等待CPU调度
　　run()：strat()调用run方法，如果实例进程时未制定传入target，这star执行t默认run()方法。
　　terminate()：不管任务是否完成，立即停止工作进程

属性：
　　daemon：和线程的setDeamon功能一样
　　name：进程名字
　　pid：进程号
```


## 多线程通信

进程是系统独立调度核分配系统资源（CPU、内存）的基本单位，进程之间是相互独立的，每启动一个新的进程相当于把数据进行了一次克隆，不同进程之间的数据互相不共享。

### 进程对列Queue

Queue是线程安全的，可以作为数据管道。
```python
from multiprocessing import Process,Queue


def fun1(q,i):
    print('子进程%s 开始put数据' %i)
    q.put('我是%s 通过Queue通信' %i)

if __name__ == '__main__':
    q = Queue()

    process_list = []
    for i in range(3):
        p = Process(target=fun1,args=(q,i,))  #注意args里面要把q对象传给我们要执行的方法，这样子进程才能和主进程用Queue来通信
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()  # 当前线程阻塞，直到p进程结束唤醒

    print('主进程获取Queue数据')
    print(q.get())
    print(q.get())
    print(q.get())
    print('结束测试')
```


### 管道Pipe
```python
from multiprocessing import Process, Pipe
def fun1(conn):
    print('子进程发送消息：')
    conn.send('你好主进程')
    mess = conn.recv()
    print(f'子进程接受消息：{mess}')
    conn.close()

if __name__ == '__main__':
    conn1, conn2 = Pipe() #关键点，pipe实例化生成一个双向管
    p = Process(target=fun1, args=(conn2,)) #conn2传给子进程
    p.start()
    mess = conn1.recv()
    print(f'主进程接受消息：{mess}')
    print('主进程发送消息：')
    conn1.send("你好子进程")
    p.join()  
    print('结束测试')
```

### Messager

Queue和Pipe只是实现了数据交互，并没实现数据共享，即一个进程去更改另一个进程的数据。那么久要用到Managers

```python
from multiprocessing import Process, Manager

def fun1(dic,lis,index):

    dic[index] = 'a'
    dic['2'] = 'b'    
    lis.append(index)    #[0,1,2,3,4,0,1,2,3,4,5,6,7,8,9]
    #print(l)

if __name__ == '__main__':
    with Manager() as manager:  # s上下文管理器，让下文的类别实现线程安全
        dic = manager.dict() #注意字典的声明方式，不能直接通过{}来定义
        l = manager.list(range(5)) #[0,1,2,3,4]

        process_list = []
        for i in range(10):
            p = Process(target=fun1, args=(dic,l,i))
            p.start()
            process_list.append(p)

        for res in process_list:
            res.join()
        print(dic)
        print(l)
```
结果
```bash
{0: 'a', '2': 'b', 3: 'a', 1: 'a', 2: 'a', 4: 'a', 5: 'a', 7: 'a', 6: 'a', 8: 'a', 9: 'a'}
[0, 1, 2, 3, 4, 0, 3, 1, 2, 4, 5, 7, 6, 8, 9]
```

## 进程池（类似于Process对象）

进程池内部维护一个进程序列，当使用时，则去进程池中获取一个进程

进程池中有两个方法：
- apply：同步，一般不使用
- apply_async：异步

```python
from  multiprocessing import Process,Pool
import os, time, random

def fun1(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    pool = Pool(2) #创建一个5个进程的进程池

    for i in range(5):
        pool.apply_async(func=fun1, args=(i,))

    pool.close()
    pool.join()
    print('结束测试')
```
输出：
```bash
Run task 0 (13216)...
Run task 1 (13217)...
Task 0 runs 1.88 seconds.
Run task 2 (13216)...
Task 1 runs 2.78 seconds.
Run task 3 (13217)...
Task 3 runs 0.40 seconds.
Run task 4 (13217)...
Task 4 runs 0.59 seconds.
Task 2 runs 1.98 seconds.
结束测试
```

> [!Note]
>  可以观察到：每次只能同时运行两个进程，一个进程结束了才能开始另一个进程。
>  
>  对`Pool`对象调用`join()`方法会等待所有子进程执行完毕，调用`join()`之前必须先调用`close()`，调用`close()`之后就不能继续添加新的`Process`了。

### 进程池map方法

直接将可迭代对象导入到函数，省得for循环。

```python
from  multiprocessing import Process,Pool
import os, time, random

def fun1(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    pool = Pool(2) #创建一个5个进程的进程池

    l = list(range(5))
    
    pool.map(fun1, l)

    pool.close()
    pool.join()
    print('结束测试')
```




# python协程

所谓python协程，其实就是`用户态线程`，在用户态之间加`锁`或者`信号量`。
> 由于python GIL锁存在，每个python程序只有一个`程序计数器PC`，因此等效于只有一个线程运行。 

> [!WARNING]
> 所以python多线程或者协程几乎只能用户IO密集型的程序，也就是说在IO的时候可以调度CPU运行其他程序。要使用机器多core性能，即CPU计算密集型的程序，还得使用python多进程 `multiprocessing`。


协程主要概念是：task, loop, 用法是`async.run()`
关键词：`async` 和 `await`，其中await 后面可以接 **协程**, **任务** 和 **Future**

## python协程使用方式

主要包含是两个模块：
- gevent：早期模块。历史：yield -> greenlet -> gevent
	- 还可以通过patch转换 thread和multiprocessing，socket，time等代码
- asyncio: 从python3.4 开始，在python3.6比较完善

![300*300](assets/img/v2-764860d01c1e1b045e9a1b01093bdb36_720w.webp)

进程内部结构

![300*300](assets/img/v2-22c9bffaeb25829cac7c3cc11b7fdb19_720w.webp)

线程内部结构，包含多个程序栈，也就是说可以同时运行多个core

而协程更进一步：直接只有一个程序栈

![500*500](assets/img/v2-05a2032eb18f9c573963fb48d060ff80_720w.webp)

示例程序：
```python
import asyncio
import aiohttp


async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()


async def main():
    urls = ['http://example.com', 'http://example.org']
    tasks = [fetch(url) for url in urls]
    responses = await asyncio.gather(*tasks)
    for response in responses:
        print(response)


asyncio.run(main())
```


> Ref: [协程大全](https://www.cnblogs.com/Amd794/p/18162269), [python高级编程](https://cloud.tencent.com/developer/article/2270470)



# 进程 协程，线程同步问题代码

## 生产者消费者问题

多进程：
```python
import multiprocessing
import random
import time
from multiprocessing import Lock

# 定义缓冲区大小
BUFFER_SIZE = 10

# 定义生产者进程
def producer(queue, producer_id, lock):
    while True:
        # 生产一个数字
        item = random.randint(1, 100)
        # 将数字放入缓冲区
        lock.acquire()
        queue.put(item)
        print(f"生产者{producer_id}生产了 {item}")
        print(f"缓冲区中还有 {queue.qsize()} 个数据")
        lock.release()
        # 随机休眠一段时间
        time.sleep(random.uniform(0.5, 2))

# 定义消费者进程
def consumer(queue, consumer_id):
    while True:
        # 从缓冲区取出一个数字
        item = queue.get()
        print(f"消费者{consumer_id}消费了 {item}")
        # 随机休眠一段时间
        time.sleep(random.uniform(0.5, 2))

def main():
    # 创建一个 multiprocessing.Queue 对象
    queue = multiprocessing.Queue(BUFFER_SIZE)
    lock = Lock()

    # 创建多个生产者进程
    producer_processes = [
        multiprocessing.Process(target=producer, args=(queue, i, lock))
        for i in range(8)
    ]
    for process in producer_processes:
        process.start()

    # 创建多个消费者进程
    consumer_processes = [
        multiprocessing.Process(target=consumer, args=(queue, i))
        for i in range(5)
    ]
    for process in consumer_processes:
        process.start()

    # 等待所有进程完成
    for process in producer_processes:
        process.join()
    for process in consumer_processes:
        process.join()

if __name__ == "__main__":
    main()
```

协程：
```python
import asyncio
import random

# 定义缓冲区大小
BUFFER_SIZE = 5

# 定义生产者协程
async def producer(queue, producer_id):
    while True:
        # 生产一个数字
        item = random.randint(1, 100)
        # 将数字放入缓冲区
        await queue.put(item)
        print(f"生产者{producer_id}生产了 {item}")
        # 随机休眠一段时间
        await asyncio.sleep(random.uniform(0.5, 2))

# 定义消费者协程
async def consumer(queue, consumer_id):
    while True:
        # 从缓冲区取出一个数字
        item = await queue.get()
        print(f"消费者{consumer_id}消费了 {item}")
        # 随机休眠一段时间
        await asyncio.sleep(random.uniform(0.5, 2))
        # 表示消费完成
        queue.task_done()

async def main():
    # 创建一个异步队列
    queue = asyncio.Queue(BUFFER_SIZE)

    # 创建多个生产者和消费者协程
    producer_tasks = [asyncio.create_task(producer(queue, i)) for i in range(3)]
    consumer_tasks = [asyncio.create_task(consumer(queue, i)) for i in range(5)]

    # 等待所有生产者和消费者协程完成
    await asyncio.gather(*producer_tasks, *consumer_tasks)

if __name__ == "__main__":
    asyncio.run(main())
```

## 吸烟者问题

假设一个系统有三个抽烟者进程和一.个供应者进程。每个抽烟者不停地卷烟并抽掉它，但是要卷起并抽掉一-支烟，抽烟者需要有三种材料:烟草、纸和胶水。三个抽烟者中，第一个拥有烟草、第二个拥有纸、第三个拥有胶水。供应者进程无限地提供三种材料，供应者每次将两种材料放桌子上，拥有剩下那种材料的抽烟者卷一 根烟并抽掉它，并给供应者进程一个信号告诉完成了，供应者就会放另外两种材料在桌上，这个过程一直重复(让三个抽烟者轮流地抽烟)

多进程：
```python
import multiprocessing
import random
import time
import os

# 定义吸烟者的种类
SMOKERS = ['paper', 'match', 'tobacco']

# 定义一个供应者进程
def agent(queue):
    while True:
        # 随机生成两种吸烟所需的材料
        materials = random.sample(SMOKERS, 2)
        # 将这两种材料放入队列
        for material in materials:
            queue.put(material)
        print(f"管理员pid: {os.getpid()}放入了 {materials[0]} 和 {materials[1]}")
        # 等待吸烟者拿走材料
        time.sleep(random.uniform(0.5, 2))

# 定义吸烟者进程
def smoker(name, queue):
    while True:
        # 从队列中取出两种材料
        material1 = queue.get()
        material2 = queue.get()
        # 检查是否有自己所需的材料
        if SMOKERS.index(material1) != SMOKERS.index(name) and SMOKERS.index(material2) != SMOKERS.index(name):
            print(f"{name} 吸烟者pid: {os.getpid()}找到了 {material1} 和 {material2}，开始吸烟")
            # 随机休眠一段时间表示吸烟
            time.sleep(random.uniform(0.5, 2))
            print(f"{name} 吸烟者pid: {os.getpid()}吸烟完毕")
        else:
            # 如果没有找到所需材料,则重新放回队列
            queue.put(material1)
            queue.put(material2)
            print(f"{name} 吸烟者pid: {os.getpid()}没有找到所需材料")
            time.sleep(random.uniform(2,4))

def main():
    # 创建一个 multiprocessing.Queue 对象
    queue = multiprocessing.Queue()

    # 创建供应者进程
    agent_process = multiprocessing.Process(target=agent, args=(queue,))
    agent_process.start()

    # 创建三个吸烟者进程
    smoker_processes = [
        multiprocessing.Process(target=smoker, args=(name, queue))
        for name in SMOKERS
    ]
    for process in smoker_processes:
        process.start()

    # 等待所有进程完成
    agent_process.join()
    for process in smoker_processes:
        process.join()

if __name__ == "__main__":
    main()
```

协程：
```python
import asyncio
import random

# 定义吸烟者的种类
SMOKERS = ['paper', 'match', 'tobacco']

# 定义一个供应者协程
async def agent(queue):
    while True:
        # 随机生成两种吸烟所需的材料
        materials = random.sample(SMOKERS, 2)
        # 将这两种材料放入队列
        for material in materials:
            await queue.put(material)
        print(f"管理员放入了 {materials[0]} 和 {materials[1]}")
        # 等待吸烟者拿走材料
        await asyncio.sleep(random.uniform(0.5, 2))

# 定义吸烟者协程
async def smoker(name, queue):
    while True:
        # 从队列中取出两种材料
        material1 = await queue.get()
        material2 = await queue.get()
        # 检查是否有自己所需的材料
        if SMOKERS.index(material1) != SMOKERS.index(name) and SMOKERS.index(material2) != SMOKERS.index(name):
            print(f"{name} 吸烟者找到了 {material1} 和 {material2}，开始吸烟")
            # 随机休眠一段时间表示吸烟
            await asyncio.sleep(random.uniform(0.5, 2))
            # 表示吸烟完成
            queue.task_done()
        else:
            # 如果没有找到所需材料,则重新放回队列
            await queue.put(material1)
            await queue.put(material2)
            print(f"{name} 吸烟者没有找到所需材料")
			await asyncio.sleep(random.uniform(2, 3)) # 让吸烟者等待原料

async def main():
    # 创建一个异步队列
    queue = asyncio.Queue()

    # 创建管理员协程
    agent_task = asyncio.create_task(agent(queue))

    # 创建三个吸烟者协程
    smoker_tasks = [asyncio.create_task(smoker(name, queue)) for name in SMOKERS]

    # 等待所有协程完成
    await asyncio.gather(agent_task, *smoker_tasks)

if __name__ == "__main__":
    asyncio.run(main())
```

## 读者写者问题

有读者和写者两组并发进程，共享一个文件，当两个或两个以上的读进程同时访问共享数据时不会产生副作用，但若某个写进程和其他进程（读进程或写进程）同时访问共享数据时则可能导致数据不一致的错误。
1.  允许多个读者可以同时对文件执行读操作
2. 只允许-一个写者 往文件中写信息
3. 任一写者在完成写操作之前不允许其他读者或写者工作


多进程：
```python
import multiprocessing
import random
import time

# 定义共享数据和锁
data = multiprocessing.Value('i', 0)
read_write_lock = multiprocessing.RLock()
read_count = multiprocessing.Value('i', 0)

# 定义读者进程
def reader(reader_id):
    global data
    while True:
        with read_write_lock:
            with read_count.get_lock():
                if read_count.value == 0:  # 第一个加锁
                    read_write_lock.acquire()
                read_count.value += 1
        print(f"读者{reader_id}正在读取数据: {data.value}")
        time.sleep(random.uniform(0.5, 2))
        with read_count.get_lock():
            read_count.value -= 1
            if read_count.value == 0:
                read_write_lock.release()  # 最后一个人解锁
        
        time.sleep(random.uniform(0.5, 2))  # 休息一下，把锁让给其他进程

# 定义写者进程
def writer(writer_id):
    global data
    while True:
        with read_write_lock:
            new_data = random.randint(1, 100)
            print(f"写者{writer_id}正在写入数据: {new_data}")
            data.value = int(new_data)
            time.sleep(random.uniform(0.5, 2))

        time.sleep(random.uniform(2, 3))  # 休息一下，把锁让给其他进程

def main():
    # 创建读者进程
    reader_processes = [
        multiprocessing.Process(target=reader, args=(i,))
        for i in range(3)
    ]
    for process in reader_processes:
        process.start()

    # 创建写者进程
    writer_processes = [
        multiprocessing.Process(target=writer, args=(i,))
        for i in range(2)
    ]
    for process in writer_processes:
        process.start()

    # 等待所有进程完成
    for process in reader_processes:
        process.join()
    for process in writer_processes:
        process.join()


if __name__ == "__main__":
    main()
```

协程：
> [!WARNING]
> 
> 协程中的锁，必须定义在，`asyncio.run()`之中，否则会出现不在loop中的错误。
> 
> 在`asyncio.run()`外部启动的`Semaphore`将获取asyncio“默认”循环，因此不能与通过`asyncio.run()`创建的事件循环一起使用。

```python
import asyncio
import random
import time

# 定义共享数据和锁
data = ""
read_count = 0

# 定义读者协程
async def reader(reader_id, read_write_lock, read_lock):
    global data, read_count
    while True:
        async with read_lock:
            read_count += 1
            if read_count == 1:
                await read_write_lock.acquire()
        print(f"out: 读者{reader_id}正在读取数据: {data}\n")
        async with read_lock:
            read_count -= 1
            if read_count == 0:
                read_write_lock.release()
        await asyncio.sleep(random.uniform(1, 3))  # 休息一下，让别人也获得锁

# 定义写者协程
async def writer(writer_id, read_write_lock):
    global data
    while True:
        async with read_write_lock:
            data = data + f"写者「{writer_id}」"
            print(f"in: 写者{writer_id}正在写入数据: {data}\n")
            await asyncio.sleep(random.uniform(0.5, 2))

async def main():
    
    read_write_lock = asyncio.Lock()
    read_lock = asyncio.Lock()
    # 创建读者协程
    reader_tasks = [
        asyncio.create_task(reader(i, read_write_lock, read_lock))
        for i in range(3)
    ]

    # 创建写者协程
    writer_tasks = [
        asyncio.create_task(writer(i, read_write_lock))
        for i in range(2)
    ]

    # 等待所有协程完成
    await asyncio.gather(*reader_tasks, *writer_tasks)

if __name__ == "__main__":
    asyncio.run(main())
```

## 哲学家进餐问题

一张圆桌上坐着5名哲学家，每两个哲学家之间的桌上摆一根筷子，桌子的中间是一碗米饭。哲学家们倾注毕生的精力用于思考和进餐，哲学家在思考时，并不影响他人。只有当哲学家饥饿时，才试图拿起左、右两根筷子（一根一根地拿起）。如果筷子已在他人手上，则需等待。饥饿的哲学家只有同时拿起两根筷子才可以开始进餐，当进餐完毕后，放下筷子继续思考。

方法：多加一把锁，让哲学家同时取到两个筷子

多进程：
```python
import multiprocessing
import time
import random
import os

# 定义餐叉
forks = [multiprocessing.Lock() for _ in range(5)]
lock = multiprocessing.Lock()

# 定义哲学家进餐进程
def philosopher(philosopher_id):
    while True:
        left_fork = forks[philosopher_id]
        right_fork = forks[(philosopher_id + 1) % 5]
        # 尝试获取左右两个餐叉，每次只能一个人拿筷子
        with lock: # 获取筷子
            left_fork.acquire()
            right_fork.acquire()
        print(f"当前pid:{os.getpid()}, 哲学家{philosopher_id}正在进餐")
        time.sleep(random.uniform(0.5, 2))  # 进餐时间
        left_fork.release()
        right_fork.release()
        
        print(f"当前pid:{os.getpid()}, 哲学家{philosopher_id}进餐完毕")
        time.sleep(random.uniform(1, 5))  # 思考时间

# 定义主进程
if __name__ == "__main__":
    # 创建5个哲学家进程
    philosopher_processes = [
        multiprocessing.Process(target=philosopher, args=(i,))
        for i in range(5)
    ]

    # 启动所有哲学家进程
    for process in philosopher_processes:
        process.start()

    # 等待所有哲学家进程结束
    for process in philosopher_processes:
        process.join()
```

协程：
```python
import asyncio
import random


# 定义哲学家进餐协程
async def philosopher(philosopher_id):
    while True:
        # 尝试获取左右两个餐叉
        async with forks[philosopher_id], forks[(philosopher_id + 1) % 5]:
            print(f"哲学家{philosopher_id}正在进餐")
            await asyncio.sleep(random.uniform(1, 5))  # 进餐时间
        print(f"哲学家{philosopher_id}进餐完毕")
        await asyncio.sleep(random.uniform(1, 5))  # 思考时间

# 定义主协程
async def main():
	# 定义餐叉
	forks = [asyncio.Lock() for _ in range(5)]
    
    # 创建5个哲学家协程
    philosopher_tasks = [
        asyncio.create_task(philosopher(i))
        for i in range(5)
    ]

    # 等待所有哲学家协程结束
    await asyncio.gather(*philosopher_tasks)

if __name__ == "__main__":
    asyncio.run(main())
```

协程：
```python
import asyncio
import random


# 定义主协程
async def main():
    # 定义餐叉
    forks = [asyncio.Lock() for _ in range(5)]
    
    # 定义哲学家进餐协程, closure
    async def philosopher(philosopher_id):
        while True:
            # 尝试获取左右两个餐叉
            async with forks[philosopher_id], forks[(philosopher_id + 1) % 5]:
                print(f"哲学家{philosopher_id}正在进餐")
                await asyncio.sleep(random.uniform(1, 5))  # 进餐时间
            print(f"哲学家{philosopher_id}进餐完毕")
            await asyncio.sleep(random.uniform(1, 5))  # 思考时间
    # 创建5个哲学家协程
    philosopher_tasks = [
        asyncio.create_task(philosopher(i))
        for i in range(5)
    ]

    # 等待所有哲学家协程结束
    await asyncio.gather(*philosopher_tasks)

if __name__ == "__main__":
    asyncio.run(main())
```