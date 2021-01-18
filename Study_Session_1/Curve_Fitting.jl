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

# ╔═╡ 45f1331e-59b1-11eb-2bc0-c138b1421b55
begin
	using Pkg
	Pkg.activate(".")
	Pkg.add("Zygote")
	Pkg.add("Plots")
	Pkg.add("PlutoUI")
end

# ╔═╡ fc1a9e20-56aa-11eb-29b1-3be8043c6ede
begin
	using Zygote
	using LinearAlgebra
	using Plots
	using PlutoUI
end

# ╔═╡ e35400cc-56ad-11eb-2794-592020c6ac7e
md"### Prediction and loss functions"

# ╔═╡ 761fc420-57af-11eb-24b3-9728b06c482e
md"We try to fit a function to a ploynomial function $f_{ML}(x)$"

# ╔═╡ 021f6506-57b1-11eb-2179-7b6f153fce2f
md"$$f_{ML}(x_i) = \displaystyle\sum\limits_{j=0}^nw_jx_i^j = w_0 + x_iw_1 + x_i^2w_2+\ldots+x_i^nw_n$$"

# ╔═╡ d2ea16ee-57b7-11eb-2d27-f3c04343dc59
md"Before we approach the real problem of solving the approximate function, we need a predicting function and loss function to evaluate good our approximation is."

# ╔═╡ bf75963a-56ac-11eb-2b34-21b724a59f96
function predict(x, ω)
	n = length(ω)
	x_vec = [x^i for i= 0:n-1]
	return dot(x_vec, ω)
end

# ╔═╡ 0129d404-57b8-11eb-3dde-efa8a687c1d4
function sq_loss(func, X, ω)
	n = length(ω)
	y = func.(X)
	ŷ = [predict(x, ω) for x in X]
	return sum((y-ŷ).^2)/n
end

# ╔═╡ 00f8133c-57be-11eb-19d0-33006b7bbb06
md"## Data Generation"

# ╔═╡ 08c5a1a6-57be-11eb-3bdc-c32957d3da8d
md"We will need a synthetic dataset for our training"

# ╔═╡ 4b28aee8-56ac-11eb-3be7-0de4e9cb211b
@bind n Slider(0:10, show_value = true, default=8)

# ╔═╡ e6d69724-57bf-11eb-0337-db41600a8ebc
md"We define a function that we ned to approximate. You can change the function and the changes will be reflected in rest of the notebook"

# ╔═╡ 9801c9ca-57bf-11eb-3f30-93055ccd6a0f
function org_func(x)
	return sin(x)
end

# ╔═╡ b2c3deb6-57b9-11eb-057f-f92ce2f4bb61
begin
	X = range(0.01, 12, length = n)
	y = org_func.(X) +0.1* randn((n))
end

# ╔═╡ 2fcaeb98-56b1-11eb-0198-a3a5088dc007
begin
	
	plot(-0.5:0.01:12, org_func.(-0.5:0.01:12), label = "sin(x)")
	scatter!(X, y, label = "data points")
end

# ╔═╡ bd151b36-56b1-11eb-03d1-094237a3bafa
md"### Approximation"

# ╔═╡ c7c2e3f2-57be-11eb-15c0-dbe72c95d654
md"We will try to solve the problem of approximating the function by two differnt ways:
- Linear Algebra Methods
- Gradient Descent using AD"

# ╔═╡ 377b50e4-57bf-11eb-1741-a73fda781579
md" ### Approximation by Linear Algebra"

# ╔═╡ 8b22ff38-57f1-11eb-3143-335475eb1cb4
md"$\begin{bmatrix}
1 & x_1 & \ldots & x_1^n\\
1 & x_2 & \ldots & x_2^n\\
1 & x_3 & \ldots & x_3^n
\end{bmatrix}$"

# ╔═╡ 371b90a6-580e-11eb-3acb-a5044f21b263
md"$\begin{bmatrix}
w_o \\
w_1 \\
\vdots\\
w_n
\end{bmatrix}$"

# ╔═╡ a5d856e6-580e-11eb-2553-758764fa82f1
md"$\begin{bmatrix}
y_1 \\
y_2 \\
\vdots\\
y_m
\end{bmatrix}$"

# ╔═╡ 9412a32e-580e-11eb-2e53-81e72bfd38b6
md"$$f_{ML}(x_i) = \displaystyle\sum\limits_{j=0}^nw_jx_i^j = w_0 + x_iw_1 + x_i^2w_2+\ldots+x_i^nw_n$$"

# ╔═╡ 4f5c808e-57bf-11eb-2a7a-3995d6f8f866
md"For now it is left to the reader"

# ╔═╡ 5dd40e02-57bf-11eb-2764-0f3f2b95517b
md" ### Approximation by Gradient Descent"

# ╔═╡ 84566232-57bf-11eb-390b-f33370fbd9a3
function train_step(func, X, ω, η)
	grad = gradient(ω->sq_loss(func, X, ω), ω)[1]
	ω -= η * grad
	return ω
end

# ╔═╡ 5c85511c-5811-11eb-0f2e-2352309f2261
ω = rand(5)

# ╔═╡ 11337388-57c1-11eb-3418-3979d91e6af2
function train(func, X, ω, η, epochs)
	for i = 1:1:epochs
		ω =  train_step(func, X, ω, η)
		loss = sq_loss(func, X, ω)
		if isnan(loss) || isinf(loss) || iszero(loss)
			break
		end
	end
	return ω
end

# ╔═╡ d722598c-57c2-11eb-3452-531fb6522e34
ω_trained = train(org_func, X, ω, 8e-9, 1e6)

# ╔═╡ 8b6bc65e-57c7-11eb-372f-bd6f700363e1
loss = sq_loss(org_func, X, ω_trained)

# ╔═╡ 328c4584-57c4-11eb-3cf5-a57e033cf574
pred = [predict(x, ω_trained) for x in -0.5:0.01:12]

# ╔═╡ a0dfe352-57c3-11eb-348f-8380ee6b3caf
begin
	plot(-0.5:0.01:15, org_func.(-0.5:0.01:15), label = "f(x)")
	scatter!(X, y, label = "Orginal data points")
	ŷ = [predict(x, ω_trained) for x in X]
	scatter!(X, ŷ, label = "Predicted data points")
	plot!(-0.5:0.01:12, pred, label = "fml(x)")
end

# ╔═╡ Cell order:
# ╠═45f1331e-59b1-11eb-2bc0-c138b1421b55
# ╠═fc1a9e20-56aa-11eb-29b1-3be8043c6ede
# ╟─e35400cc-56ad-11eb-2794-592020c6ac7e
# ╟─761fc420-57af-11eb-24b3-9728b06c482e
# ╟─021f6506-57b1-11eb-2179-7b6f153fce2f
# ╟─d2ea16ee-57b7-11eb-2d27-f3c04343dc59
# ╠═bf75963a-56ac-11eb-2b34-21b724a59f96
# ╠═0129d404-57b8-11eb-3dde-efa8a687c1d4
# ╟─00f8133c-57be-11eb-19d0-33006b7bbb06
# ╟─08c5a1a6-57be-11eb-3bdc-c32957d3da8d
# ╠═4b28aee8-56ac-11eb-3be7-0de4e9cb211b
# ╟─e6d69724-57bf-11eb-0337-db41600a8ebc
# ╠═9801c9ca-57bf-11eb-3f30-93055ccd6a0f
# ╠═b2c3deb6-57b9-11eb-057f-f92ce2f4bb61
# ╠═2fcaeb98-56b1-11eb-0198-a3a5088dc007
# ╟─bd151b36-56b1-11eb-03d1-094237a3bafa
# ╟─c7c2e3f2-57be-11eb-15c0-dbe72c95d654
# ╟─377b50e4-57bf-11eb-1741-a73fda781579
# ╟─8b22ff38-57f1-11eb-3143-335475eb1cb4
# ╟─371b90a6-580e-11eb-3acb-a5044f21b263
# ╟─a5d856e6-580e-11eb-2553-758764fa82f1
# ╟─9412a32e-580e-11eb-2e53-81e72bfd38b6
# ╟─4f5c808e-57bf-11eb-2a7a-3995d6f8f866
# ╟─5dd40e02-57bf-11eb-2764-0f3f2b95517b
# ╠═84566232-57bf-11eb-390b-f33370fbd9a3
# ╠═5c85511c-5811-11eb-0f2e-2352309f2261
# ╠═11337388-57c1-11eb-3418-3979d91e6af2
# ╠═d722598c-57c2-11eb-3452-531fb6522e34
# ╠═8b6bc65e-57c7-11eb-372f-bd6f700363e1
# ╠═328c4584-57c4-11eb-3cf5-a57e033cf574
# ╠═a0dfe352-57c3-11eb-348f-8380ee6b3caf
