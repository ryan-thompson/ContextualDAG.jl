ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = "1"

using ContextualDAG
using Test

const Flux = ContextualDAG.Flux

@testset "ContextualDAG.jl" begin
    @testset "Flux compatibility" begin
        x = zeros(2, 1)
        error = try
            cdag(x, x, x, x; initialise = "invalid", hidden_layers = [1], verbose = false)
            nothing
        catch error
            error
        end
        @test error isa ErrorException
        @test occursin("initialise", sprint(showerror, error))

        model = ContextualDAG.gennet([3, 2], 2, 1, Flux.relu)
        @test size(model(rand(Float32, 1, 4))) == (2, 2, 4)

        model = ContextualDAG.gennet(Int[], 2, 1, Flux.relu)
        @test size(model(rand(Float32, 1, 4))) == (2, 2, 4)
    end

    @testset "batched CPU operations" begin
        x = Float32[1 2; 3 4]
        w = reshape(Float32.(1:8), 2, 2, 2)
        expected = hcat(transpose(w[:, :, 1]) * x[:, 1],
            transpose(w[:, :, 2]) * x[:, 2])
        @test ContextualDAG.xw_mult(w; x = x) == expected
        gradient = Flux.gradient(w -> sum(ContextualDAG.xw_mult(w; x = x)), w)[1]
        @test gradient == reshape(x, 2, 1, 2) .* ones(Float32, 1, 2, 2)

        A = [Float32[2 0; 0 4], Float32[1 1; 0 2]]
        C = [similar(A_i) for A_i in A]
        ContextualDAG.matinv_batched!(A, C)
        @test C[1] ≈ inv(A[1])
        @test C[2] ≈ inv(A[2])
    end

    @testset "fit and predict" begin
        x = Float32[1 0; 0 1; 1 1; -1 1; 0.5 -0.5; -0.5 0.5]
        z = reshape(Float32[-1, 0, 1, -0.5, 0.5, 1.5], :, 1)
        fit = cdag(x, z, x, z; lambda = [0f0], hidden_layers = [2], epoch_max = 1,
            patience = 1, optimiser = () -> Flux.Adam(0.01), verbose = false,
            params = (1, 1, 0.5, 1f-2, 1, 1, 0.1, 0.1))

        w = coef(fit, Float32[-0.25; 0.25;;])
        @test size(w) == (2, 2, 2)
        @test all(isfinite, w)
        @test all(iszero, [w[i, i, k] for i in 1:2, k in 1:2])
    end
end
