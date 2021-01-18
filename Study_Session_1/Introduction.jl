using LinearAlgebra
# using <package name>

print("Hello World")

println("Hello World")

for i=range(1, 10, step = 1)
    println(i)
end

function square(a::Int)
    return a^2
end

function foo(a)
    a = "asdasd"
    a = 3
    a = 5.6
end

function foo(a::Int)
    a^3
end

foo(4)

foo(6.7)

square(4)

square([1, 2, 3, 4])

square.([1, 2, 3, 4])