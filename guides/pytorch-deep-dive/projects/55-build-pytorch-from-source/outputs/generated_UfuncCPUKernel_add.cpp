#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/ufunc/add.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/Dispatch.h>
#include <c10/core/Scalar.h>

namespace at {
namespace native {

namespace {

void add_kernel(TensorIteratorBase& iter, const at::Scalar & alpha) {
  AT_DISPATCH_SWITCH(iter.common_dtype(), "add_stub",

AT_DISPATCH_CASE(at::ScalarType::Bool,
  [&]() {

auto _s_alpha = alpha.to<scalar_t>();
cpu_kernel(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); }
);

  }
)


AT_DISPATCH_CASE(at::ScalarType::Byte,
  [&]() {

auto _s_alpha = alpha.to<scalar_t>();
auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);

  }
)


AT_DISPATCH_CASE(at::ScalarType::Char,
  [&]() {

auto _s_alpha = alpha.to<scalar_t>();
auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);

  }
)


AT_DISPATCH_CASE(at::ScalarType::Int,
  [&]() {

auto _s_alpha = alpha.to<scalar_t>();
auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);

  }
)


AT_DISPATCH_CASE(at::ScalarType::Long,
  [&]() {

auto _s_alpha = alpha.to<scalar_t>();
auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);

  }
)


AT_DISPATCH_CASE(at::ScalarType::Short,
  [&]() {

auto _s_alpha = alpha.to<scalar_t>();
auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);

  }
)


AT_DISPATCH_CASE(at::ScalarType::Float,
  [&]() {

auto _s_alpha = alpha.to<scalar_t>();
auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);

  }
)


AT_DISPATCH_CASE(at::ScalarType::Double,
  [&]() {

auto _s_alpha = alpha.to<scalar_t>();
auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);

  }
)


AT_DISPATCH_CASE(at::ScalarType::ComplexFloat,
  [&]() {

auto _s_alpha = alpha.to<scalar_t>();
auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);

  }
)


AT_DISPATCH_CASE(at::ScalarType::ComplexDouble,
  [&]() {

auto _s_alpha = alpha.to<scalar_t>();
auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);

  }
)


AT_DISPATCH_CASE(at::ScalarType::BFloat16,
  [&]() {

auto _s_alpha = alpha.to<scalar_t>();
auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);

  }
)


AT_DISPATCH_CASE(at::ScalarType::Half,
  [&]() {

auto _s_alpha = alpha.to<scalar_t>();
auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);

  }
)


AT_DISPATCH_CASE(at::ScalarType::ComplexHalf,
  [&]() {

auto _s_alpha = alpha.to<scalar_t>();
auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);

  }
)

  );
}

} // anonymous namespace

using add_fn = void(*)(TensorIteratorBase&, const at::Scalar &);
DECLARE_DISPATCH(add_fn, add_stub)
REGISTER_DISPATCH(add_stub, &add_kernel)
}} // namespace at::native
