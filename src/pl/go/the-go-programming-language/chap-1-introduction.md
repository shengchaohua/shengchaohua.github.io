---
title: 第 1 章 入门
order: 1
---

## 1.1. Hello, World
代码如下：
```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, 世界")
}
```

Go 是一门编译型语言。

Go 语言提供了 go 命令以及一系列子命令。

go run 命令可以编译一个或多个以 .go 结尾的源文件，并运行最终生成的可执行文件。
```shell
go run helloworld.go
```
