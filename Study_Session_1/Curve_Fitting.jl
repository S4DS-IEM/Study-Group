### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ fc1a9e20-56aa-11eb-29b1-3be8043c6ede
begin
	using Zygote
	using LinearAlgebra
	using Plots
	using PlutoUI
	using LaTeXStrings
	using ImageCore
end

# ╔═╡ 45f1331e-59b1-11eb-2bc0-c138b1421b55
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
end

# ╔═╡ 60a23536-59d0-11eb-368a-5588b71a6093
md"# Curve fitting using AD"

# ╔═╡ 832e2c5a-59ce-11eb-32d7-85b05440f919
md"## Problem Formulation"

# ╔═╡ e35400cc-56ad-11eb-2794-592020c6ac7e
md"### The underlying mathematics of the problem"

# ╔═╡ 761fc420-57af-11eb-24b3-9728b06c482e
md"We are trying to fit a function $f_{org}(x)$ to a ploynomial function $f_{ML}(x)$"

# ╔═╡ 021f6506-57b1-11eb-2179-7b6f153fce2f
md"$$f_{ML}(x_i) = \displaystyle\sum\limits_{j=0}^nw_jx_i^j = w_0 + x_iw_1 + x_i^2w_2+\ldots+x_i^nw_n$$"

# ╔═╡ df0e42cc-59be-11eb-367e-cf25ab5f105f
md"so that $f_{org}(x_i) \approx f_{ML}(x_i)$"

# ╔═╡ 2730398c-59bf-11eb-060e-276c0138395e
md"We define $X, \textbf{w}, \textbf{y}$ for a more concise and all-round understanding of the problem,"

# ╔═╡ e50c3cc4-59b7-11eb-1479-8bba33d251fd
md"$\begin{gather}
X=
\begin{bmatrix}
1 & x_1 & \ldots & x_1^n\\
\vdots & & \ddots &\vdots \\
1 & x_m & & x_m^n
\end{bmatrix}
\quad\quad\quad
\textbf{w} = \begin{bmatrix}
w_o \\
w_1 \\
\vdots\\
w_n
\end{bmatrix}
\quad\quad\quad
\textbf{y} = 
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots\\
y_m
\end{bmatrix}
\end{gather}$"

# ╔═╡ 51545508-59b9-11eb-3f94-59e743bfd1a8
md" where,
- The matrix $X$ is the data matrix. $X$ is a $(m\times (n+1)) matrix$. $m$ is the number of data points and $n$ is the n is the degree of the plyonomial. The choice of this polynomial becomes clear in the next step.
- The vector $\textbf{w}$ represents the weights vector. $w_0$ the bias. $w_j$ is the weight corresponding to $x^j$.
- The vector $\textbf{y}$ represents the results vector. $y_i$ is the result of $x_i$ on applying the function $f$"

# ╔═╡ 70b0cea0-59ba-11eb-2ae9-439069c94467
md"$\begin{gather}
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots\\
y_m
\end{bmatrix}
=
\begin{bmatrix}
1 & x_1 & \ldots & x_1^n\\
\vdots & & \ddots &\vdots \\
1 & x_m & & x_m^n
\end{bmatrix}
\odot 
\begin{bmatrix}
w_o \\
w_1 \\
\vdots\\
w_n
\end{bmatrix}
\end{gather}$"

# ╔═╡ ea14bfa4-59ba-11eb-35a1-6548f64bed7e
md"$\begin{gather}
\quad \quad \
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots\\
y_m
\end{bmatrix}
=
\begin{bmatrix}
w_o + x_1\times w_1 &\ldots  &+x_1^n \times w_n\\
\vdots \\
w_0 + x_m\times w_1  &\ldots &+x_m^n\times w_n
\end{bmatrix}
\end{gather}$"

# ╔═╡ 32bb7c2c-59bc-11eb-0914-d99847d081a3
load(download("https://thehritikjain.files.wordpress.com/2020/05/1_e4_ptjctmaofsrpzczbv-g-1.jpeg?w=935"))

# ╔═╡ d2ea16ee-57b7-11eb-2d27-f3c04343dc59
md"Before we approach the real problem of solving the approximate function, we need a predicting function and loss function to evaluate good our approximation is."

# ╔═╡ fa7fe6b4-59c1-11eb-2486-d90418eaee4a
md"### Predict and loss functions : "

# ╔═╡ e8864a02-59c1-11eb-12ff-c9a4ffde849d
md"#### Predict function "

# ╔═╡ bf75963a-56ac-11eb-2b34-21b724a59f96
function predict(x, ω::Vector)
	n = length(ω)
	x_vec = [x^i for i= 0:n-1]
	return dot(x_vec, ω)
end

# ╔═╡ c8a25b68-59c1-11eb-363f-0772ec4edbf2
md"**Usage Example :**"

# ╔═╡ 10be275c-59c1-11eb-10dc-4becfaf27f8d
begin
	# Try changing the values
	local ω = rand(5) # 4 polynomial weight vector
	local x = 3. # input
	predict(x, ω)
end	

# ╔═╡ d928b270-59c1-11eb-09d0-a9c3c878fe2d
md"#### Square Loss function "

# ╔═╡ 0129d404-57b8-11eb-3dde-efa8a687c1d4
function sq_loss(func, X, ω::Vector) 
	n = length(ω)
	y = func.(X)
	ŷ = [predict(x, ω) for x in X]
	return sum((y-ŷ).^2)/n
end

# ╔═╡ bc758f7c-59c1-11eb-3f4a-5b987c7c6e0b
md"**Usage Example :**"

# ╔═╡ 55bd943c-59c1-11eb-19d3-a7bba6fcbd30
begin
	local ω = rand(5) # 4 polynomial weight vector
	local X = [3.] # input
	local loss = sq_loss(sin, X, ω)
end

# ╔═╡ 00f8133c-57be-11eb-19d0-33006b7bbb06
md"## Data Generation"

# ╔═╡ 6d407f0e-59ce-11eb-0ed6-33f817c914a3
md"Choose the number of data points : "

# ╔═╡ 4b28aee8-56ac-11eb-3be7-0de4e9cb211b
@bind n Slider(0:10, show_value = true, default=8)

# ╔═╡ e6d69724-57bf-11eb-0337-db41600a8ebc
md"We define a function that we want to approximate. **org\_func** is the original function. We are trying to approximate *sin(x)* here. You can try your own function that you want to approximat inside org\_func and chnages will be reflected in rest of the notebook."

# ╔═╡ 9801c9ca-57bf-11eb-3f30-93055ccd6a0f
function org_func(x)
	return sin(x)
end

# ╔═╡ 2fcaeb98-56b1-11eb-0198-a3a5088dc007
begin
	X = [x for x in range(randn(), 12, length = n)]
	y = org_func.(X) +0.1* randn((n))
	plot(-0.5:0.01:12, org_func.(-0.5:0.01:12), label = L"sin(x)")
	scatter!(X, y, label = "data points", title = "Our Data points")
end

# ╔═╡ bd151b36-56b1-11eb-03d1-094237a3bafa
md"## Approximation"

# ╔═╡ c7c2e3f2-57be-11eb-15c0-dbe72c95d654
md"We will try to solve the problem of approximating the function by two differnt ways:
- Linear Algebra Methods
- Gradient Descent using AD"

# ╔═╡ 377b50e4-57bf-11eb-1741-a73fda781579
md" ### Approximation by Linear Algebra"

# ╔═╡ 50c00e44-59c4-11eb-2726-27bee50e4fdd
md"$\begin{gather}
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots\\
y_m
\end{bmatrix}
=
\begin{bmatrix}
1 & x_1 & \ldots & x_1^n\\
\vdots & & \ddots &\vdots \\
1 & x_m & & x_m^n
\end{bmatrix}
\odot 
\begin{bmatrix}
w_o \\
w_1 \\
\vdots\\
w_n
\end{bmatrix}
\end{gather}$"

# ╔═╡ 586240fe-59c4-11eb-1427-312c47374f8a
md"$
\ \ \ \
\begin{gather}
\begin{bmatrix}
w_o \\
w_1 \\
\vdots\\
w_n
\end{bmatrix}
=
\begin{bmatrix}
1 & x_1 & \ldots & x_1^n\\
\vdots & & \ddots &\vdots \\
1 & x_m & & x_m^n
\end{bmatrix}^{-1}
\odot 
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots\\
y_m
\end{bmatrix}

\end{gather}$"

# ╔═╡ 4f5c808e-57bf-11eb-2a7a-3995d6f8f866
md"This may seem like a very viable method, but try considering the case where $X$ is invertible. It is left as an exercise to the reader."

# ╔═╡ 5dd40e02-57bf-11eb-2764-0f3f2b95517b
md" ### Approximation by Gradient Descent"

# ╔═╡ ea205094-59c9-11eb-0f04-818f051b7d1e
md"We will be using gradient descent algorithm to find $\textbf{w}$ here. We are tired of mathematics by now and we somehow want the differentaition to happen automatically(almost magically). We will be using **Automatic Differentation**(or a more intuitive name : **Algorthimic Differentiation**). While how automatic differintation works is another day's agenda, we can have a little taste of magic of what AD does for us."

# ╔═╡ b13969a4-59ca-11eb-15d8-0f9b5f225fed
md"The gradient of a function $f(x)$ at $x_0$ is defined as $\displaystyle\frac{df(x)}{dx}\Bigg|_{x=x_0}$

$f^\prime(x_0) = \displaystyle\frac{df(x)}{dx}\Bigg|_{x=x_0}$"

# ╔═╡ e4f40500-59cb-11eb-1020-3b345469fb1f
md"The Zygote package gives us a *gradient* API that we can use to find gradients"

# ╔═╡ 88002972-59cc-11eb-3d75-ab7e7bb45954
md"$f(x) = x^2+4x+1$
$f^\prime(x) = 2x+4$
$f^\prime(5) = 2\times 5 + 4 = 14$"

# ╔═╡ 3255da08-59cc-11eb-0bdc-61a05344ad50
gradient(x->x^2+4x+1,  5)

# ╔═╡ cd6a28f8-59cc-11eb-28f8-35c3a04414d8
md"$f(x, y) = x^2+y^2+3xy$
$\displaystyle\frac{\partial f(x,y)}{\partial x} = 2x+3y$
$\displaystyle\frac{\partial f(x,y)}{\partial y} = 2y+3x$"

# ╔═╡ 4506b2d0-59cc-11eb-22e2-5f43abeb3ccd
gradient( (x,y) -> x^2 + y^2 + 3x*y, 2,3)

# ╔═╡ 608cc19c-59cd-11eb-092d-558d322966c6
md"A train step function that returns the updated weights vector $\textbf{w}$"

# ╔═╡ 84566232-57bf-11eb-390b-f33370fbd9a3
function train_step(func, X, ω, η)::typeof(ω)
	grad = gradient(ω->sq_loss(func, X, ω), ω)[1]
	ω -= η * grad
	return ω
end

# ╔═╡ de0aeb62-59cd-11eb-3e5f-a7cb176093c9
md"Choose the degree of polynomial of  $f_{ML}(x)$. You may need to choose different $\eta$ and number of epochs to account for the model change"

# ╔═╡ ae1c0e18-59cd-11eb-2e61-fbe810bd5fa0
@bind n_model Slider(1:1:7, default = 4, show_value = true)

# ╔═╡ 5c85511c-5811-11eb-0f2e-2352309f2261
## Intializing weights
ω = rand(n_model+1)

# ╔═╡ 11337388-57c1-11eb-3418-3979d91e6af2
function train(func, X, ω, η, epochs)::typeof(ω)
	for i = 1:1:epochs
		ω =  train_step(func, X, ω, η)
		loss = sq_loss(func, X, ω)
		if isnan(loss) || isinf(loss) || iszero(loss)
			throw(DomainError("Loss cannot take $(loss) value"))
		end
	end
	return ω
end

# ╔═╡ d722598c-57c2-11eb-3452-531fb6522e34
ω_trained = train(org_func, X, ω, 7e-9, 1e6)

# ╔═╡ 11d38860-59c6-11eb-06e2-b10697f21f1d
md"It is upto the user to decide how much accuracy he wants given the time constraints"

# ╔═╡ 8b6bc65e-57c7-11eb-372f-bd6f700363e1
loss = sq_loss(org_func, X, ω_trained)

# ╔═╡ 328c4584-57c4-11eb-3cf5-a57e033cf574
pred = [predict(x, ω_trained) for x in -0.5:0.01:13]

# ╔═╡ a0dfe352-57c3-11eb-348f-8380ee6b3caf
begin
	plot(-0.5:0.01:13, org_func.(-0.5:0.01:13), label = L"f_{org}(x)")
	scatter!(X, y, label = "Orginal data points")
	ŷ = [predict(x, ω_trained) for x in X]
	scatter!(X, ŷ, label = "Predicted data points")
	plot!(-0.5:0.01:13, pred, label = L"f_{ML}(x)")
end

# ╔═╡ Cell order:
# ╟─fc1a9e20-56aa-11eb-29b1-3be8043c6ede
# ╟─45f1331e-59b1-11eb-2bc0-c138b1421b55
# ╟─60a23536-59d0-11eb-368a-5588b71a6093
# ╟─832e2c5a-59ce-11eb-32d7-85b05440f919
# ╟─e35400cc-56ad-11eb-2794-592020c6ac7e
# ╟─761fc420-57af-11eb-24b3-9728b06c482e
# ╟─021f6506-57b1-11eb-2179-7b6f153fce2f
# ╟─df0e42cc-59be-11eb-367e-cf25ab5f105f
# ╟─2730398c-59bf-11eb-060e-276c0138395e
# ╟─e50c3cc4-59b7-11eb-1479-8bba33d251fd
# ╟─51545508-59b9-11eb-3f94-59e743bfd1a8
# ╟─70b0cea0-59ba-11eb-2ae9-439069c94467
# ╟─ea14bfa4-59ba-11eb-35a1-6548f64bed7e
# ╟─32bb7c2c-59bc-11eb-0914-d99847d081a3
# ╟─d2ea16ee-57b7-11eb-2d27-f3c04343dc59
# ╟─fa7fe6b4-59c1-11eb-2486-d90418eaee4a
# ╟─e8864a02-59c1-11eb-12ff-c9a4ffde849d
# ╠═bf75963a-56ac-11eb-2b34-21b724a59f96
# ╟─c8a25b68-59c1-11eb-363f-0772ec4edbf2
# ╠═10be275c-59c1-11eb-10dc-4becfaf27f8d
# ╟─d928b270-59c1-11eb-09d0-a9c3c878fe2d
# ╠═0129d404-57b8-11eb-3dde-efa8a687c1d4
# ╟─bc758f7c-59c1-11eb-3f4a-5b987c7c6e0b
# ╠═55bd943c-59c1-11eb-19d3-a7bba6fcbd30
# ╟─00f8133c-57be-11eb-19d0-33006b7bbb06
# ╟─6d407f0e-59ce-11eb-0ed6-33f817c914a3
# ╟─4b28aee8-56ac-11eb-3be7-0de4e9cb211b
# ╟─e6d69724-57bf-11eb-0337-db41600a8ebc
# ╠═9801c9ca-57bf-11eb-3f30-93055ccd6a0f
# ╠═2fcaeb98-56b1-11eb-0198-a3a5088dc007
# ╟─bd151b36-56b1-11eb-03d1-094237a3bafa
# ╟─c7c2e3f2-57be-11eb-15c0-dbe72c95d654
# ╟─377b50e4-57bf-11eb-1741-a73fda781579
# ╟─50c00e44-59c4-11eb-2726-27bee50e4fdd
# ╟─586240fe-59c4-11eb-1427-312c47374f8a
# ╟─4f5c808e-57bf-11eb-2a7a-3995d6f8f866
# ╟─5dd40e02-57bf-11eb-2764-0f3f2b95517b
# ╟─ea205094-59c9-11eb-0f04-818f051b7d1e
# ╟─b13969a4-59ca-11eb-15d8-0f9b5f225fed
# ╟─e4f40500-59cb-11eb-1020-3b345469fb1f
# ╟─88002972-59cc-11eb-3d75-ab7e7bb45954
# ╠═3255da08-59cc-11eb-0bdc-61a05344ad50
# ╟─cd6a28f8-59cc-11eb-28f8-35c3a04414d8
# ╠═4506b2d0-59cc-11eb-22e2-5f43abeb3ccd
# ╟─608cc19c-59cd-11eb-092d-558d322966c6
# ╠═84566232-57bf-11eb-390b-f33370fbd9a3
# ╟─de0aeb62-59cd-11eb-3e5f-a7cb176093c9
# ╟─ae1c0e18-59cd-11eb-2e61-fbe810bd5fa0
# ╠═5c85511c-5811-11eb-0f2e-2352309f2261
# ╠═11337388-57c1-11eb-3418-3979d91e6af2
# ╠═d722598c-57c2-11eb-3452-531fb6522e34
# ╟─11d38860-59c6-11eb-06e2-b10697f21f1d
# ╠═8b6bc65e-57c7-11eb-372f-bd6f700363e1
# ╠═328c4584-57c4-11eb-3cf5-a57e033cf574
# ╟─a0dfe352-57c3-11eb-348f-8380ee6b3caf
