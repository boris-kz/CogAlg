using Images, ImageView, CSV, Tables, IterTools, DataFrames

# instead of the class in Python version in Julia values are stored in the struct
mutable struct Cdert_
    i::Int
    p::Int
    d::Int
    m::Real # Both Int and Float types
    mrdn::Bool
end

mutable struct CP
    L::Int
    I::Int
    D::Int
    M::Real # summed ave - abs(d), different from D
    Rdn::Int  # 1 + binary dert.mrdn cnt / len(dert_)
    x0::Int
    dert_::Vector{Cdert_}  # contains (i, p, d, m, mrdn)
    subset::Any # 1st sublayer' rdn, rng, xsub_pmdertt_, _xsub_pddertt_, sub_Ppm_, sub_Ppd_
    # for layer-parallel access and comp, ~ frequency domain, composition: 1st: dert_, 2nd: sub_P_[ dert_], 3rd: sublayers[ sub_P_[ dert_]]:
    sublayers::Vector{Any}  # multiple layers of sub_P_s from d segmentation or extended comp, nested to depth = sub_[n]
end

# pattern filters or hyper-parameters: eventually from higher-level feedback, initialized here as constants:
ave = 15  # |difference| between pixels that coincides with average value of Pm
ave_min = 2  # for m defined as min |d|: smaller?
ave_M = 20  # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 5  # min |D| for initial incremental-derivation comparison(d_)


function deriv_incr_P_(rootP, P_, rdn, rng)
    # First, convert P_ to a DataFrame.
    df_P_ = DataFrame(P_)
    # Add the other arguments to the DataFrame.
    n = size(df_P_, 1) # number of rows in df_P_

    # Handle the case if rootP is Nothing; otherwise, create a repeated vector
    rootP_column = isnothing(rootP) ? fill(missing, n) : fill(rootP, n)

    # Add the other arguments to the DataFrame.
    df_P_[!, :rootP] = rootP_column
    df_P_[!, :rdn] = fill(rdn, n)
    df_P_[!, :rng] = fill(rng, n)

    # CSV.write("./layer3_log_jl.csv", P_, append=true)
    # Write the DataFrame to the CSV file, appending it.
    CSV.write("./layer3_log_jl.csv", df_P_, append=true)

    comb_sublayers = []  # The same notation
    for (id, P) in enumerate(P_)
        if abs(P.D) - (P.L - P.Rdn) * ave_D * P.L > ave_D * rdn && P.L > 1  # high-D span, ave_adj_M is represented in ave_D
            rdn += 1
            rng += 1
            push!(P.subset, [rdn, rng, [], [], [], []])  # 1st sublayer params, []s: xsub_pmdertt_, _xsub_pddertt_, sub_Ppm_, sub_Ppd_
            sub_Pm_, sub_Pd_ = ([], [])  # initialize layers, concatenate by intra_P_ in form_P_
            push!(P.sublayers, (sub_Pm_, sub_Pd_))  # 1st layer
            ddert_ = []
            _d = abs(P.dert_[1].d) #! differs from python, starting with 1
            for (id2, dert) in enumerate(P.dert_[2:end])  # all same-sign in Pd
                d = abs(dert.d)  # compare ds
                rd = d + _d
                dd = d - _d
                md = min(d, _d) - abs(dd / 2) - ave_min  # min_match because magnitude of derived vars corresponds to predictive value
                # md = convert(Int64, round(min(d, _d) - abs(dd) / 2 - ave_min)) #! Different from Python, type conversion required
                dmrdn = md < 0
                push!(ddert_, Cdert_(dert.d, rd, dd, md, dmrdn))
                _d = d
            end
            # sub_Pm_[:] .= form_P_(P, ddert_, rdn=rdn, rng=rng, fPd=false)  # cluster by mm sign
            # sub_Pd_[:] .= form_P_(P, ddert_, rdn=rdn, rng=rng, fPd=true)  # cluster by md sign
            push!(sub_Pm_, form_P_(P, ddert_, rdn=rdn, rng=rng, fPd=false))  # cluster by mm sign
            push!(sub_Pd_, form_P_(P, ddert_, rdn=rdn, rng=rng, fPd=true))  # cluster by md sign
        end

        if !isnothing(rootP) && !isnothing(P.sublayers) && length(comb_sublayers) > 0
            new_comb_sublayers = []
            for ((comb_sub_Pm_, comb_sub_Pd_), (sub_Pm_, sub_Pd_)) in zip_longest(comb_sublayers, P.sublayers, ([], []))
                comb_sub_Pm_ .+= sub_Pm_  # remove brackets, they preserve index in sub_Pp root_
                comb_sub_Pd_ .+= sub_Pd_
                push!(new_comb_sublayers, (comb_sub_Pm_, comb_sub_Pd_))  # add sublayer
            end
            comb_sublayers = new_comb_sublayers
        end
    end

    if !isnothing(rootP)
        # rootP.sublayers .= rootP.sublayers .+ comb_sublayers
        push!(rootP.sublayers, comb_sublayers)
    end
end


function form_P_(rootP, dert_; rdn, rng, fPd)  # after semicolon in Julia keyword args are placed
    # initialization:
    P_ = CP[]  # structure to store all the form_P (layer1) output
    x = 0
    _sign = nothing  # to initialize 1st P, (nothing != True) and (nothing != False) are both True

    # for dert in dert_  # segment by sign
    for (dert_id, dert) in enumerate(dert_)  # segment by sign
        if fPd == true
            sign = dert.d > 0
        else
            sign = dert.m > 0
        end

        if sign != _sign
            # sign change, initialize and append P
            L = 1
            I = dert.p
            D = dert.d
            M = dert.m
            Rdn = dert.mrdn + 1
            x0 = x
            sublayers = [] # Rdn starts from 1
            push!(P_, CP(L, I, D, M, Rdn, x0, [dert], [], []))  # save data in the struct
        else
            # accumulate params:
            P_[end].L += 1
            P_[end].I += dert.p
            P_[end].D += dert.d
            P_[end].M += dert.m
            P_[end].Rdn += dert.mrdn
            push!(P_[end].dert_, dert)
        end
        x += 1
        _sign = sign
    end

    # range_incr_P_(rootP, P_, rdn, rng)
    deriv_incr_P_(rootP, P_, rdn, rng)

    if fPd == false
        CSV.write("./layer2_Pm_log_jl.csv", P_, append=true)
    else
        CSV.write("./layer2_Pd_log_jl.csv", P_, append=true)
    end

    return P_  # used only if not rootP, else packed in rootP.sublayers
end

run_mode = 1

if run_mode == 1
    # Read the CSV file into a DataFrame
    df = CSV.File("layer1_log_jl.csv") |> DataFrame

    # Extract the first row into a Cdert_ instance
    function row_to_struct(row)
        return Cdert_(row.i, row.p, row.d, row.m, row.mrdn)
    end

    # Convert all rows to Cdert_ instances and store in the list dert_
    dert_ = [row_to_struct(df[i, :]) for i in 1:1023]

    # parameter_names = ["L" "I" "D" "M" "Rdn" "x0" "dert_" "subset" "sublayers"]  # Vector
    # CSV.write("./layer2_Pd_log_jl.csv", Tables.table(parameter_names), writeheader=false)
    # CSV.write("./layer2_Pm_log_jl.csv", Tables.table(parameter_names), writeheader=false)
    parameter_names2 = ["L" "I" "D" "M" "Rdn" "x0" "dert_" "subset" "sublayers" "rootP" "rdn" "rng"]  # Vector
    CSV.write("./layer3_log_jl.csv", Tables.table(parameter_names2), writeheader=false)

    # # form patterns, evaluate them for rng+ and der+ sub-recursion of cross_comp:
    Pm_ = form_P_(nothing, dert_, rdn=1, rng=1, fPd=false)  # rootP=None, eval intra_P_ (calls form_P_)
    Pd_ = form_P_(nothing, dert_, rdn=1, rng=1, fPd=true)
end

if run_mode == 2
    parameter_names = ["L" "I" "D" "M" "Rdn" "x0" "dert_" "subset" "sublayers"]  # Vector
    CSV.write("./layer3_log_jl.csv", Tables.table(parameter_names), writeheader=false)

    P_ = CP[]
    # dert = Cdert_[Cdert_(179, 355, 3, 12, false), Cdert_(183, 362, 4, 11, false), Cdert_(186, 369, 3, 12, false), Cdert_(188, 374, 2, 13, false)]
    dert = Cdert_(179, 355, 3, 12, false)
    push!(P_, CP(4, 1460, 12, 48, 1, 1019, [dert], [], []))  # save data in the struct
    println(deriv_incr_P_(nothing, P_, 1, 1))
end