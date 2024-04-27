---
title: 第 1 章 初识 Kafka
order: 1
---


## 前言

Kafka 有三大用途：

- 消息系统：具有系统解耦、冗余存储、流量削峰、缓冲、异步通信、扩展性、可恢复性等功能。
- 存储系统：把消息持久化到磁盘，相比于其他基于内存存储的系统而言，降低了数据丢失的风险。
- 流式处理平台：Kafka 提供了一个完整的流式处理类库，比如窗口、连接、变换和聚合等各类操作。



## 1.1 基本概念

一个典型的 Kafka 体系架构包含若干 Producer、若干 Broker、若干 Consumer，以及一个 ZooKeeper 集群。

-  ZooKeeper 是 Kafka 用来负责集群元数据的管理、控制器的选举等操作的。
- Producer 将消息发送到 Broker。
- Broker 负责将收到的消息存储到磁盘中。
- Consumer 负责从 Broker 订阅并消费消息。

如下图所示：

![Kafka 体系结构](https://raw.githubusercontent.com/shengchaohua/my-images/main/images/202404272135061.png)

整个 Kafka 体系结构引入了以下 3 个术语：

（1）Producer：生产者。负责创建消息，然后将其投递到 Kafka 中。

（2）Consumer：消费者。消费者连接到 Kafka 上并接收消息，进而进行相应的业务逻辑处理。

（3）Broker：服务代理节点。对于 Kafka 而言，Broker 可以简单地看作一个独立的 Kafka 服务节点或 Kafka 服务实例。一个或多个 Broker 组成了一个 Kafka 集群。一般而言，我们更习惯使用首字母小写的 broker 来表示服务代理节点。

另外还有两个特别重要的概念：主题（Topic）和分区（Partition）。Kafka 中的消息以主题为单位进行归类，生产者负责将消息发送到特定的主题（发送到 Kafka 集群中的每一个消息都要指定一个主题），而消费者负责订阅主题并进行消费。

主题是一个逻辑上的概念，它还可以细分为多个分区，一个分区只属于一个主题，很多时候也会把分区称为主题分区（Topic-Partition）。

- 同一主题下的不同分区包含的消息是不同的，分区在存储层面可以看作一个可追加的日志（Log）文件。
- 消息在被追加到分区日志文件的时候都会分配一个特定的偏移量（offset）。
- offset 是消息在分区中的唯一标识，Kafka 通过它来保证消息在分区内的顺序性。offset 并不跨越分区，也就是说，Kafka 保证的是分区有序而不是主题有序。

- 分区可以分布在不同的服务器（broker）上。也就是说，一个主题可以跨越多个 broker，以此来提供比单个 broker 更强大的性能。

每一条消息被发送到 broker 之前，会根据分区规则选择存储到哪个具体的分区。

- 如果分区规则设定得合理，所有消息就可以均匀地分配到不同的分区。
- 如果一个主题只有一个分区，那么这个分区对应的日志文件所在的机器 IO 将会成为这个主题的性能瓶颈，而分区解决了这个问题。
- 在创建主题的时候可以设置分区的个数，当然也可以在主题创建完成之后去增加分区的数量，实现水平扩展。

Kafka 为分区引入了多副本（Replica）机制，通过增加副本数量可以提升容灾能力。如下图所示，

![多副本架构](https://raw.githubusercontent.com/shengchaohua/my-images/main/images/202404272136309.png)

解释：

- 副本之间是“一主多从”的关系，其中 leader 副本负责对生产者和消费者提供服务，处理读写请求；follower 副本只负责与 leader 副本的消息同步，所以 follower 副本中的消息相对 leader 副本会有一定的滞后。
- 副本位于不同的 broker 中，当 leader 副本出现故障时，Kafka 会从 follower 副本中重新选举新的 leader 副本对外提供服务。
- Kafka 通过多副本机制实现了故障的自动转移，当 Kafka 集群中某个 broker 失效时，仍然能保证服务可用。

分区中的所有副本统称为 AR（Assigned Replicas）。

- 所有与 leader 副本保持一定程度同步的副本（包括 leader 副本在内）组成 ISR（In-Sync Replicas），其中“一定程度”是指可以忍受的滞后范围，可以通过参数进行配置。ISR 集合是 AR 集合中的一个子集，
- 消息会先发送到 leader 副本，然后 follower 副本才能从 leader 副本中拉去消息进行同步，同步期间 follower 副本会有一定程度的滞后。与 leader 副本同步滞后过多的副本（不包括 leader 副本）组成 OSR（Out-of-Sync Replicas）。由此可见，AR=IR+OSR。
- 在正常情况下，所有的 follower 副本都应该与 leader 副本保持一定程度的同步，即 AR=ISR，OSR 集合为空。
- leader 副本负责维护和跟踪 ISR 集合中所有 follower 副本的滞后状态：
  - 如果 follower 副本落后太多或失效时，leader 副本会把它从 ISR 集合中剔除。
  - 如果 OSR 集合中有 follower 副本“追上”了 leader 副本，那么 leader 副本会把它从 OSR 集合转移到 ISR 集合中。
  - 默认情况下，当 leader 副本发生故障时，只有 ISR 集合中的副本才有资格被选举为新的 leader。

ISR 集合与 HW 和 LEO 有密切的关系。

- HW 是 High Watermark，标识了一个特定的消息偏移量（offset），意味着消费者只能拉取到这个 offset 之前的消息。
- LEO 是 Log End Offset，标识了一个日志文件中下一条待写入消息的 offset。每个 offset 对应一条消息，LEO 的大小相当于当前日志文件中最后一条消息的 offset 值加 1。举例，如果一个日志文件中，最后一条消息的 offset 为 8，那么 LEO 等于 9。
- 分区 ISR 集合中的每个副本都会维护自身的 LEO，而 ISR 集合中最小的 LEO 即为分区的 HW。
- LEO 是副本维度的数据，HW 是分区维度的数据。



## 1.2 安装与配置

略。



## 1.3 生产和消费

略。



## 1.4 服务端参数配置

Kafka 服务端有很多参数配置，下面介绍一些重要的参数。

1）zookeeper.connect

该参数指明 broker 要连接的 ZooKeeper 集群地址（包含端口号），没有默认值，且为必填项。

2）listeners

该参数指明 broker 监听客户端连接的地址列表，即为客户端要连接 broker 的入口地址列表。

3）broker.id

该参数用来指定 Kafka 集群中 broker 的唯一标识，默认值为 -1。如果没有设置，那么 Kafka 会自动生成一个。

4）log.dir 和 log.dirs

Kafka 把所有的消息都保存在磁盘上，而这两个参数用来配置日志文件存放的根目录。一般情况下，log.dir 用来配置单个根目录，log.dirs 用来配置多个根目录，后者的优先级更高。默认情况下只配置了 log.dir 参数，其默认值为 /tmp/kafka-logs。

5）message.max.bytes

该参数用来指定 broker 所能接受消息的最大值，默认值为 1000012，约等于 976.6 KB。如果 Producer 发送的消息大于这个参数所设置的值，那么 Producer 就会出错。