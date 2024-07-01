#include <cassert>
#include <optional>

#include <pybind11/stl.h>
#include <torch/torch.h>

namespace py = pybind11;

torch::Tensor sparse_linear_forward(
    torch::Tensor input,
    torch::Tensor lora_B,
    torch::Tensor lora_A,
    torch::Tensor dv,
    torch::Tensor di,
    std::optional<torch::Tensor> bias
) {
    torch::Tensor W = lora_B.mm(lora_A.to(lora_B.dtype()));
    W.reshape(-1).scatter_add_(0, di.to(torch::kInt64), dv.to(W.dtype()));
    return torch::nn::functional::linear(
        input.to(W.dtype()),
        W,
        bias ? *bias : torch::Tensor()
    );
}



torch::autograd::tensor_list sparse_linear_backward(
    torch::Tensor output_grad,
    torch::Tensor input,
    torch::Tensor lora_B,
    torch::Tensor lora_A,
    torch::Tensor dv,
    torch::Tensor di,
    bool input_needs_grad,
    bool lora_B_needs_grad,
    bool lora_A_needs_grad,
    bool dv_needs_grad,
    bool bias_needs_grad,
    std::optional<torch::Tensor> bias = std::nullopt
) {
    torch::Tensor W = lora_B.mm(lora_A.to(lora_B.dtype()));
    di = di.to(torch::kInt64);
    W.reshape(-1).scatter_add_(0, di, dv.to(W.dtype()));

    torch::Tensor input_grad, lora_B_grad, lora_A_grad, dv_grad, bias_grad;
    torch::Tensor output_grad_2d = output_grad.reshape({-1, output_grad.size(-1)});

    if (input_needs_grad)
        input_grad = output_grad_2d.mm(W.to(output_grad_2d.dtype())).view_as(input);

    torch::Tensor input_2d = input.reshape({-1, input.size(-1)});
    torch::Tensor weight_grad = output_grad_2d.t().mm(input_2d.to(output_grad_2d.dtype()));

    if (lora_B_needs_grad)
        lora_B_grad = weight_grad.mm(lora_A.t().to(weight_grad.dtype()));

    if (lora_A_needs_grad)
        lora_A_grad = lora_B.t().mm(weight_grad.to(lora_B.dtype()));

    if (dv_needs_grad)
        dv_grad = weight_grad.view(-1).gather(0, di);

    if (bias && bias_needs_grad)
        bias_grad = output_grad_2d.sum(0);

    return {input_grad, lora_B_grad, lora_A_grad, dv_grad, torch::Tensor(), bias_grad};
}



class SparseLinearME : public torch::autograd::Function<SparseLinearME> {
  public:

    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor input,
        torch::Tensor lora_B,
        torch::Tensor lora_A,
        torch::Tensor dv,
        torch::Tensor di,
        std::optional<torch::Tensor> bias
    ) {
        if (bias)
            ctx->save_for_backward({input, lora_B, lora_A, dv, di, *bias});
        else
            ctx->save_for_backward({input, lora_B, lora_A, dv, di});
        return sparse_linear_forward(input, lora_B, lora_A, dv, di, bias);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto lora_B = saved[1];
        auto lora_A = saved[2];
        auto dv = saved[3];
        auto di = saved[4];
        auto output_grad = grad_outputs[0];

        return sparse_linear_backward(
            output_grad,
            input,
            lora_B,
            lora_A,
            dv,
            di,
            ctx->needs_input_grad(0),
            ctx->needs_input_grad(1),
            ctx->needs_input_grad(2),
            ctx->needs_input_grad(3),
            saved.size() > 5 && ctx->needs_input_grad(5),
            saved.size() > 5 ?
                std::optional<torch::Tensor>(saved[5]) :
                std::optional<torch::Tensor>(std::nullopt)
        );
    }

};

torch::Tensor apply_sparse_linear(
    torch::Tensor input,
    torch::Tensor lora_B,
    torch::Tensor lora_A,
    torch::Tensor dv,
    torch::Tensor di,
    std::optional<torch::Tensor> bias = std::nullopt
) {
    return SparseLinearME::apply(
        input,
        lora_B,
        lora_A,
        dv,
        di,
        bias
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &sparse_linear_forward
    );
    m.def(
        "backward",
        &sparse_linear_backward
    );
    m.def(
        "apply",
        &apply_sparse_linear,
        py::arg("input"),
        py::arg("lora_B"),
        py::arg("lora_A"),
        py::arg("dv"),
        py::arg("di"),
        py::arg_v("bias", std::nullopt)
    );
}
