// Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/testkit/test_kit.h>
#include <vespa/eval/eval/tensor_function.h>
#include <vespa/eval/eval/simple_tensor.h>
#include <vespa/eval/eval/simple_tensor_engine.h>
#include <vespa/eval/tensor/default_tensor_engine.h>
#include <vespa/eval/tensor/dense/dense_tensor.h>
#include <vespa/eval/eval/test/tensor_model.hpp>
#include <vespa/eval/eval/test/eval_fixture.h>

#include <vespa/vespalib/util/stringfmt.h>
#include <vespa/vespalib/util/stash.h>

using namespace vespalib;
using namespace vespalib::eval;
using namespace vespalib::eval::test;
using namespace vespalib::tensor;
using namespace vespalib::eval::tensor_function;

const TensorEngine &prod_engine = DefaultTensorEngine::ref();

double seq_value = 0.0;

struct GlobalSequence : public Sequence {
    GlobalSequence() {}
    double operator[](size_t) const override {
        seq_value += 1.0;
        return seq_value;
    }
    ~GlobalSequence() {}
};
GlobalSequence seq;

EvalFixture::ParamRepo make_params() {
    return EvalFixture::ParamRepo()
        .add("con_x5_A", spec({x(5)}, seq))
        .add("con_x5_B", spec({x(5)}, seq))
        .add("con_x5_C", spec({x(5)}, seq))
        .add("con_x5y3_A", spec({x(5),y(3)}, seq))
        .add("con_x5y3_B", spec({x(5),y(3)}, seq))
        .add_mutable("mut_dbl_A", spec(1.5))
        .add_mutable("mut_dbl_B", spec(2.5))
        .add_mutable("mut_x5_A", spec({x(5)}, seq))
        .add_mutable("mut_x5_B", spec({x(5)}, seq))
        .add_mutable("mut_x5_C", spec({x(5)}, seq))
        .add_mutable("mut_x5f_D", spec(float_cells({x(5)}), seq))
        .add_mutable("mut_x5f_E", spec(float_cells({x(5)}), seq))
        .add_mutable("mut_x5y3_A", spec({x(5),y(3)}, seq))
        .add_mutable("mut_x5y3_B", spec({x(5),y(3)}, seq))
        .add_mutable("mut_x_sparse", spec({x({"a", "b", "c"})}, seq));
}
EvalFixture::ParamRepo param_repo = make_params();

void verify_optimized(const vespalib::string &expr, size_t param_idx) {
    EvalFixture fixture(prod_engine, expr, param_repo, true, true);
    EXPECT_EQUAL(fixture.result(), EvalFixture::ref(expr, param_repo));
    for (size_t i = 0; i < fixture.num_params(); ++i) {
        TEST_STATE(vespalib::make_string("param %zu", i).c_str());
        if (i == param_idx) {
            EXPECT_EQUAL(fixture.get_param(i), fixture.result());
        } else {
            EXPECT_NOT_EQUAL(fixture.get_param(i), fixture.result());
        }
    }
}

void verify_p0_optimized(const vespalib::string &expr) {
    verify_optimized(expr, 0);
}

void verify_p1_optimized(const vespalib::string &expr) {
    verify_optimized(expr, 1);
}

void verify_p2_optimized(const vespalib::string &expr) {
    verify_optimized(expr, 2);
}

void verify_not_optimized(const vespalib::string &expr) {
    EvalFixture fixture(prod_engine, expr, param_repo, true, true);
    EXPECT_EQUAL(fixture.result(), EvalFixture::ref(expr, param_repo));
    for (size_t i = 0; i < fixture.num_params(); ++i) {
        EXPECT_NOT_EQUAL(fixture.get_param(i), fixture.result());
    }
}

TEST("require that mutable dense concrete tensors are optimized") {
    TEST_DO(verify_p1_optimized("mut_x5_A-mut_x5_B"));
    TEST_DO(verify_p0_optimized("mut_x5_A-con_x5_B"));
    TEST_DO(verify_p1_optimized("con_x5_A-mut_x5_B"));
    TEST_DO(verify_p1_optimized("mut_x5y3_A-mut_x5y3_B"));
    TEST_DO(verify_p0_optimized("mut_x5y3_A-con_x5y3_B"));
    TEST_DO(verify_p1_optimized("con_x5y3_A-mut_x5y3_B"));
}

TEST("require that self-join operations can be optimized") {
    TEST_DO(verify_p0_optimized("mut_x5_A+mut_x5_A"));
}

TEST("require that join(tensor,scalar) operations are not optimized") {
    TEST_DO(verify_not_optimized("mut_x5_A-mut_dbl_B"));
    TEST_DO(verify_not_optimized("mut_dbl_A-mut_x5_B"));
}

TEST("require that join with different tensor shapes are optimized") {
    TEST_DO(verify_p1_optimized("mut_x5_A*mut_x5y3_B"));
}

TEST("require that inplace join operations can be chained") {
    TEST_DO(verify_p2_optimized("mut_x5_A+(mut_x5_B+mut_x5_C)"));
    TEST_DO(verify_p0_optimized("(mut_x5_A+con_x5_B)+con_x5_C"));
    TEST_DO(verify_p1_optimized("con_x5_A+(mut_x5_B+con_x5_C)"));
    TEST_DO(verify_p2_optimized("con_x5_A+(con_x5_B+mut_x5_C)"));
}

TEST("require that non-mutable tensors are not optimized") {
    TEST_DO(verify_not_optimized("con_x5_A+con_x5_B"));
}

TEST("require that scalar values are not optimized") {
    TEST_DO(verify_not_optimized("mut_dbl_A+mut_dbl_B"));
    TEST_DO(verify_not_optimized("mut_dbl_A+5"));
    TEST_DO(verify_not_optimized("5+mut_dbl_B"));
}

TEST("require that mapped tensors are not optimized") {
    TEST_DO(verify_not_optimized("mut_x_sparse+mut_x_sparse"));
}

TEST("require that optimization works with float cells") {
    TEST_DO(verify_p1_optimized("mut_x5f_D-mut_x5f_E"));
}

TEST("require that overwritten value must have same cell type as result") {
    TEST_DO(verify_p0_optimized("mut_x5_A-mut_x5f_D"));
    TEST_DO(verify_p1_optimized("mut_x5f_D-mut_x5_A"));
    TEST_DO(verify_not_optimized("con_x5_A-mut_x5f_D"));
    TEST_DO(verify_not_optimized("mut_x5f_D-con_x5_A"));
}

TEST_MAIN() { TEST_RUN_ALL(); }
