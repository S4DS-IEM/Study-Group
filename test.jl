using LinearAlgebra

print("Hello World")

println()

for i=1:1:10
    println(i)
end

function square(a::Int)
    return a^2
end

square([1, 2, 3, 4])

a = square.([1, 2, 3, 4 ])