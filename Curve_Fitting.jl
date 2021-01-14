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
end

# ╔═╡ e35400cc-56ad-11eb-2794-592020c6ac7e
md"### Predict function"

# ╔═╡ bf75963a-56ac-11eb-2b34-21b724a59f96
function predict(x, ω)
	n = length(ω)
	x_vec = [x^i for i=1:n]
	return dot(x_vec, ω)
end

# ╔═╡ d3301456-56ad-11eb-3417-4df631af7f9c
md"### Squared loss function"

# ╔═╡ 68d98ce0-56ad-11eb-134e-6393d0871ca5
function square_loss(y, ŷ)
	return (y-ŷ)^2
end

# ╔═╡ 02fe4e70-56ae-11eb-3a26-19fbb3372cdb
md" ## Time to train"

# ╔═╡ 4b28aee8-56ac-11eb-3be7-0de4e9cb211b
@bind n Slider(0:10, show_value = true, default=8)

# ╔═╡ b91fe286-56ac-11eb-2d2e-4973bc823280
ω = rand(n+1)

# ╔═╡ 87d3491a-56ad-11eb-11a1-1b20693c1856
gradient(ω->square_loss(sin(1), predict(1, ω)), ω)

# ╔═╡ 5a545ce2-56b0-11eb-261e-d5ecd0e356e4
data = rand(1:0.01:12, (n))

# ╔═╡ 028000a6-56b1-11eb-1c70-57344246cb8b
y = sin.(data) +0.2* randn((n))

# ╔═╡ 2fcaeb98-56b1-11eb-0198-a3a5088dc007
begin
	
	plot(1:0.01:12, sin.(1:0.01:12), label = "sin(x)")
	scatter!(data, y, label = "data points")
end

# ╔═╡ bd151b36-56b1-11eb-03d1-094237a3bafa


# ╔═╡ Cell order:
# ╠═fc1a9e20-56aa-11eb-29b1-3be8043c6ede
# ╟─e35400cc-56ad-11eb-2794-592020c6ac7e
# ╠═bf75963a-56ac-11eb-2b34-21b724a59f96
# ╟─d3301456-56ad-11eb-3417-4df631af7f9c
# ╠═68d98ce0-56ad-11eb-134e-6393d0871ca5
# ╟─02fe4e70-56ae-11eb-3a26-19fbb3372cdb
# ╠═4b28aee8-56ac-11eb-3be7-0de4e9cb211b
# ╠═b91fe286-56ac-11eb-2d2e-4973bc823280
# ╠═87d3491a-56ad-11eb-11a1-1b20693c1856
# ╠═5a545ce2-56b0-11eb-261e-d5ecd0e356e4
# ╠═028000a6-56b1-11eb-1c70-57344246cb8b
# ╠═2fcaeb98-56b1-11eb-0198-a3a5088dc007
# ╠═bd151b36-56b1-11eb-03d1-094237a3bafa
