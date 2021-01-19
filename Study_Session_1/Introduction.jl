# using <package name>
using Random

# Print without new line
print("Hello World")

# Print with new line
println("Hello World")

for i in range(1, 10, step = 1)
    println(i)
end

function square(a::Int)
    return a*a
end

# Call the function
print(square(6))

# Vectorization 
vec_sq = square.([1, 2, 3, 4])



# Mapping
mapped_vec = map(x->square(x)+4, [1,2,3,4])