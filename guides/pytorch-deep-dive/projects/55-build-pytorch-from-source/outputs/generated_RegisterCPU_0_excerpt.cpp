// Excerpt of the generated RegisterCPU_0.cpp (10338 lines total),
// produced locally by `python -m torchgen.gen`. Lines 1201-1320.
// The dispatcher reported add.Tensor's CPU kernel at /pytorch/build/aten/src/ATen/RegisterCPU_0.cpp:1309

  1201          if (!names.empty()) {
  1202            namedinference::propagate_names(outputs_[output_idx], names);
  1203          }
  1204          // super must happen after, so that downstream can use maybe_get_output
  1205          // to retrieve the output
  1206          at::native::structured_ufunc_add_CPU::set_output_raw_strided(output_idx, sizes, strides, options, names);
  1207      }
  1208      const Tensor& maybe_get_output(int64_t output_idx) override {
  1209        return outputs_[output_idx];
  1210      }
  1211      std::array<Tensor, 1> outputs_;
  1212  };
  1213  at::Tensor wrapper_CPU_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  1214  structured_ufunc_add_CPU_functional op;
  1215  op.meta(self, other, alpha);
  1216  op.impl(self, other, alpha, op.outputs_[0]);
  1217  return std::move(op.outputs_[0]);
  1218  }
  1219  struct structured_ufunc_add_CPU_out final : public at::native::structured_ufunc_add_CPU {
  1220      structured_ufunc_add_CPU_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
  1221      void set_output_strided(
  1222          int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
  1223          TensorOptions options, DimnameList names
  1224      ) override {
  1225          const auto& out = outputs_[output_idx].get();
  1226          resize_out(out, sizes, strides, options);
  1227          auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
  1228          if (C10_UNLIKELY(maybe_proxy.has_value())) {
  1229              proxy_outputs_[output_idx] = std::move(maybe_proxy).value();
  1230          }
  1231          if (!names.empty()) {
  1232            namedinference::propagate_names(outputs_[output_idx], names);
  1233          }
  1234          // super must happen after, so that downstream can use maybe_get_output
  1235          // to retrieve the output
  1236          at::native::structured_ufunc_add_CPU::set_output_raw_strided(output_idx, sizes, strides, options, names);
  1237      }
  1238      void set_output_raw_strided(
  1239          int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
  1240          TensorOptions options, DimnameList names
  1241      ) override {
  1242          const auto& out = outputs_[output_idx].get();
  1243          resize_out(out, sizes, strides, options);
  1244          if (!names.empty()) {
  1245            namedinference::propagate_names(outputs_[output_idx], names);
  1246          }
  1247          // super must happen after, so that downstream can use maybe_get_output
  1248          // to retrieve the output
  1249          at::native::structured_ufunc_add_CPU::set_output_raw_strided(output_idx, sizes, strides, options, names);
  1250      }
  1251      const Tensor& maybe_get_output(int64_t output_idx) override {
  1252        return proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get();
  1253      }
  1254      std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  1255      std::array<::std::optional<Tensor>, 1> proxy_outputs_;
  1256  };
  1257  at::Tensor & wrapper_CPU_add_out_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  1258  structured_ufunc_add_CPU_out op(out);
  1259  op.meta(self, other, alpha);
  1260  op.impl(self, other, alpha, op.maybe_get_output(0));
  1261  if (op.proxy_outputs_[0].has_value()) op.outputs_[0].get().copy_(*op.proxy_outputs_[0]);
  1262  return out;
  1263  }
  1264  struct structured_ufunc_add_CPU_inplace final : public at::native::structured_ufunc_add_CPU {
  1265      structured_ufunc_add_CPU_inplace(Tensor& self) : outputs_{std::ref(self)} {}
  1266      void set_output_strided(
  1267          int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
  1268          TensorOptions options, DimnameList names
  1269      ) override {
  1270          const auto& out = outputs_[output_idx].get();
  1271          check_inplace(out, sizes, options);
  1272          auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
  1273          if (C10_UNLIKELY(maybe_proxy.has_value())) {
  1274              proxy_outputs_[output_idx] = std::move(maybe_proxy).value();
  1275          }
  1276          if (!names.empty()) {
  1277            namedinference::propagate_names(outputs_[output_idx], names);
  1278          }
  1279          // super must happen after, so that downstream can use maybe_get_output
  1280          // to retrieve the output
  1281          at::native::structured_ufunc_add_CPU::set_output_raw_strided(output_idx, sizes, strides, options, names);
  1282      }
  1283      void set_output_raw_strided(
  1284          int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
  1285          TensorOptions options, DimnameList names
  1286      ) override {
  1287          const auto& out = outputs_[output_idx].get();
  1288          check_inplace(out, sizes, options);
  1289          if (!names.empty()) {
  1290            namedinference::propagate_names(outputs_[output_idx], names);
  1291          }
  1292          // super must happen after, so that downstream can use maybe_get_output
  1293          // to retrieve the output
  1294          at::native::structured_ufunc_add_CPU::set_output_raw_strided(output_idx, sizes, strides, options, names);
  1295      }
  1296      const Tensor& maybe_get_output(int64_t output_idx) override {
  1297        return proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get();
  1298      }
  1299      std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  1300      std::array<::std::optional<Tensor>, 1> proxy_outputs_;
  1301  };
  1302  at::Tensor & wrapper_CPU_add__Tensor(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  1303  structured_ufunc_add_CPU_inplace op(self);
  1304  op.meta(self, other, alpha);
  1305  op.impl(self, other, alpha, op.outputs_[0]);
  1306  if (op.proxy_outputs_[0].has_value()) op.outputs_[0].get().copy_(*op.proxy_outputs_[0]);
  1307  return self;
  1308  }
  1309  TORCH_LIBRARY_IMPL(aten, CPU, m) {
  1310      m.impl("add.Tensor", TORCH_FN(wrapper_CPU_add_Tensor));
  1311  m.impl("add.out", TORCH_FN(wrapper_CPU_add_out_out));
  1312  m.impl("add_.Tensor", TORCH_FN(wrapper_CPU_add__Tensor));
  1313  }
  1314  } // anonymous namespace
  1315  namespace cpu {
  1316  at::Tensor add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  1317  return wrapper_CPU_add_Tensor(self, other, alpha);
  1318  }
  1319  at::Tensor & add_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  1320  return wrapper_CPU_add_out_out(self, other, alpha, out);
