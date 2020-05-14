// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/testkit/test_kit.h>
#include <vespa/eval/eval/tensor_function.h>
#include <vespa/eval/eval/simple_tensor.h>
#include <vespa/eval/eval/simple_tensor_engine.h>
#include <vespa/eval/tensor/default_tensor_engine.h>
#include <vespa/eval/tensor/dense/dense_simple_join_function.h>
#include <vespa/eval/eval/test/eval_fixture.h>
#include <vespa/eval/eval/test/tensor_model.hpp>

#include <vespa/vespalib/util/stringfmt.h>

using namespace vespalib;
using namespace vespalib::eval;
using namespace vespalib::eval::test;
using namespace vespalib::tensor;
using namespace vespalib::eval::tensor_function;

using vespalib::make_string_short::fmt;

using Primary = DenseSimpleJoinFunction::Primary;
using Overlap = DenseSimpleJoinFunction::Overlap;

std::ostream &operator<<(std::ostream &os, Primary primary)
{
    switch(primary) {
    case Primary::LHS: return os << "LHS";
    case Primary::RHS: return os << "RHS";
    }
    abort();
}

std::ostream &operator<<(std::ostream &os, Overlap overlap)
{
    switch(overlap) {
    case Overlap::FULL: return os << "FULL";
    case Overlap::INNER: return os << "INNER";
    case Overlap::OUTER: return os << "OUTER";
    }
    abort();
}

const TensorEngine &prod_engine = DefaultTensorEngine::ref();

EvalFixture::ParamRepo make_params() {
    return EvalFixture::ParamRepo()
        .add("a", spec(1.5))
        .add("b", spec(2.5))
        .add("sparse", spec({y({"a"})}, N()))
        .add("mixed", spec({x(5),y({"a"})}, N()))
        .add_vector("x", 5, [](size_t idx){ return double((idx * 2) + 3); })
        .add_vector("x", 5, [](size_t idx){ return double((idx * 3) + 2); });
}
EvalFixture::ParamRepo param_repo = make_params();

void verify_optimized(const vespalib::string &expr, Primary primary, Overlap overlap, bool pri_mut, size_t factor, int p_inplace = -1) {
    EvalFixture slow_fixture(prod_engine, expr, param_repo, false);
    EvalFixture fixture(prod_engine, expr, param_repo, true, true);
    EXPECT_EQUAL(fixture.result(), EvalFixture::ref(expr, param_repo));
    EXPECT_EQUAL(fixture.result(), slow_fixture.result());
    auto info = fixture.find_all<DenseSimpleJoinFunction>();
    ASSERT_EQUAL(info.size(), 1u);
    EXPECT_TRUE(info[0]->result_is_mutable());
    EXPECT_EQUAL(info[0]->primary(), primary);
    EXPECT_EQUAL(info[0]->overlap(), overlap);
    EXPECT_EQUAL(info[0]->primary_is_mutable(), pri_mut);
    EXPECT_EQUAL(info[0]->factor(), factor);
    if (p_inplace >= 0) {
        ASSERT_TRUE(fixture.num_params() > size_t(p_inplace));
        EXPECT_EQUAL(fixture.get_param(p_inplace), fixture.result());
    }
}

void verify_not_optimized(const vespalib::string &expr) {
    EvalFixture slow_fixture(prod_engine, expr, param_repo, false);
    EvalFixture fixture(prod_engine, expr, param_repo, true);
    EXPECT_EQUAL(fixture.result(), EvalFixture::ref(expr, param_repo));
    EXPECT_EQUAL(fixture.result(), slow_fixture.result());
    auto info = fixture.find_all<DenseSimpleJoinFunction>();
    EXPECT_TRUE(info.empty());
}

TEST("require that basic join is optimized") {
    TEST_DO(verify_optimized("x5+x5$2", Primary::RHS, Overlap::FULL, false, 1));
}

TEST("require that subset join with complex overlap is not optimized") {
}

TEST("require that scalar values are not optimized") {
    TEST_DO(verify_not_optimized("a+b"));
    TEST_DO(verify_not_optimized("a+x5"));
    TEST_DO(verify_not_optimized("x5+b"));
    TEST_DO(verify_not_optimized("a+sparse"));
    TEST_DO(verify_not_optimized("sparse+a"));
    TEST_DO(verify_not_optimized("a+mixed"));
    TEST_DO(verify_not_optimized("mixed+a"));
}

TEST("require that mapped tensors are not optimized") {
    TEST_DO(verify_not_optimized("sparse+sparse"));
    TEST_DO(verify_not_optimized("sparse+x5"));
    TEST_DO(verify_not_optimized("x5+sparse"));
    TEST_DO(verify_not_optimized("sparse+mixed"));
    TEST_DO(verify_not_optimized("mixed+sparse"));
}

TEST("require mixed tensors are not optimized") {
    TEST_DO(verify_not_optimized("mixed+mixed"));
    TEST_DO(verify_not_optimized("mixed+x5"));
    TEST_DO(verify_not_optimized("x5+mixed"));
}

TEST_MAIN() { TEST_RUN_ALL(); }
