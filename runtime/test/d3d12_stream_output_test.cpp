// D3D-302: a StreamOutput PSO and SOSetTargets actually capture a draw's vertex output on the Vulkan-backed
// device. vs_stream_output.vert writes the fixed vec4 (1, 2, 3, 4) to an Xfb-decorated output (buffer 0, stride
// 16 bytes); the PSO declares one matching StreamOutput buffer slot. After a 3-vertex draw, the buffer is read
// back and every vertex's 16 bytes are checked against that constant, for RasterizedStream 0 (the ordinary case)
// and D3D12_SO_NO_RASTERIZED_STREAM (capture-only, rasterizerDiscardEnable): both must still capture correctly.
// No game data; needs VK_EXT_transform_feedback.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)
#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

using namespace d3d12test;

namespace {

// Builds a StreamOutput PSO with the given RasterizedStream, draws 3 vertices through it, and checks that the
// buffer SOSetTargets bound captured 3 copies of vs_stream_output.vert's fixed vec4 (1, 2, 3, 4).
bool run_case(ID3D12Device* device, ID3D12RootSignature* root, D3D12_CPU_DESCRIPTOR_HANDLE rtv,
             const std::vector<std::uint8_t>& vs_container, const std::vector<std::uint8_t>& ps_container,
             UINT width, UINT height, UINT rasterized_stream, const char* label) {
    const UINT stride = 16;
    const D3D12_SO_DECLARATION_ENTRY so_entry{0, "OUT", 0, 0, 4, 0};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root;
    pso_desc.VS = {vs_container.data(), vs_container.size()};
    pso_desc.PS = {ps_container.data(), ps_container.size()};
    pso_desc.StreamOutput.pSODeclaration = &so_entry;
    pso_desc.StreamOutput.NumEntries = 1;
    pso_desc.StreamOutput.pBufferStrides = &stride;
    pso_desc.StreamOutput.NumStrides = 1;
    pso_desc.StreamOutput.RasterizedStream = rasterized_stream;
    pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso_desc.SampleMask = 0xffffffff;
    pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso_desc.RasterizerState.DepthClipEnable = TRUE;
    pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pso_desc.SampleDesc.Count = 1;
    Ref<ID3D12PipelineState> pso;
    CHECK(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

    // The SO buffer (DEFAULT heap, created directly in STREAM_OUT state) and a READBACK copy of it.
    constexpr UINT64 so_bytes = 256;
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC buf_desc{};
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Width = so_bytes;
    buf_desc.Height = 1;
    buf_desc.DepthOrArraySize = 1;
    buf_desc.MipLevels = 1;
    buf_desc.SampleDesc.Count = 1;
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Ref<ID3D12Resource> so_buffer;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &buf_desc,
                                                    D3D12_RESOURCE_STATE_STREAM_OUT, nullptr,
                                                    IID_PPV_ARGS(&so_buffer.p))));
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> so_readback;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buf_desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&so_readback.p))));

    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                              IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    const D3D12_VIEWPORT viewport{0, 0, float(width), float(height), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(width), LONG(height)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->SetGraphicsRootSignature(root);
    list->SetPipelineState(pso.p);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_STREAM_OUTPUT_BUFFER_VIEW so_view{so_buffer->GetGPUVirtualAddress(), so_bytes, 0};
    list->SOSetTargets(0, 1, &so_view);
    list->DrawInstanced(3, 1, 0, 0);
    const auto to_copy_src = transition(so_buffer.p, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy_src);
    list->CopyBufferRegion(so_readback.p, 0, so_buffer.p, 0, so_bytes);
    CHECK(SUCCEEDED(list->Close()));

    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    const HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    CHECK(ev != nullptr);
    const bool done = SUCCEEDED(fence->SetEventOnCompletion(1, ev)) && WaitForSingleObject(ev, 20000) == WAIT_OBJECT_0;
    CloseHandle(ev);
    CHECK(done);

    void* mapped = nullptr;
    CHECK(SUCCEEDED(so_readback->Map(0, nullptr, &mapped)) && mapped);
    const auto* floats = static_cast<const float*>(mapped);
    int mismatches = 0;
    for (int vertex = 0; vertex < 3; ++vertex) {
        const float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        for (int component = 0; component < 4; ++component) {
            const float got = floats[vertex * 4 + component];
            if (got != expected[component]) {
                std::fprintf(stderr, "%s: vertex %d component %d: got %f, expected %f\n", label, vertex, component,
                            double(got), double(expected[component]));
                ++mismatches;
            }
        }
    }
    const D3D12_RANGE no_write{0, 0};
    so_readback->Unmap(0, &no_write);
    CHECK(mismatches == 0);
    std::printf("stream_output %s: 3 vertices captured, %d mismatches\n", label, mismatches);
    return true;
}

}  // namespace

int main() {
    vktest::Device vk;
    vk.request_transform_feedback = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(status == vktest::DeviceStatus::ready);
    if (!vk.transform_feedback) {
        std::fprintf(stderr, "SKIP: no VK_EXT_transform_feedback on this device\n");
        return 77;
    }

    // Shader programs and the index that maps their hashes to the compiled test modules (fssm_sha256 is a
    // lookup key the test controls on both sides; it has no relation to the real shader bytes).
    const auto vs_program = fssm_program({0x56530302u, 0x00000001u});
    const auto ps_program = fssm_program({0x50530302u, 0x00000001u});
    const auto vs_container = shader_container(vs_program);
    const auto ps_container = shader_container(ps_program);
    const auto fixture = std::filesystem::temp_directory_path() /
                         ("leonida_stream_output_test_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(fixture);
    std::filesystem::copy_file(LEONIDA_TEST_VS_STREAM_OUTPUT_SPV, fixture / "vs_stream_output.spv",
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(LEONIDA_TEST_PS_STREAM_OUTPUT_SPV, fixture / "ps_stream_output.spv",
                               std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(fixture / "index.json", std::ios::binary);
        index << "{\"vs_stream_output\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 0, \"input_signature\": []},\n"
                 " \"ps_stream_output\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 0, \"outputs\": [0]}}";
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = true;
    info.subgroup = vk.subgroup;
    info.shader_dir = fixture.string();
    info.transform_feedback = vk.transform_feedback;
    info.transform_feedback_geometry_streams = vk.transform_feedback_geometry_streams;
    info.max_transform_feedback_buffers = vk.max_transform_feedback_buffers;
    info.max_transform_feedback_streams = vk.max_transform_feedback_streams;
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

    // Empty root signature: neither shader reads a root parameter.
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto root_blob = serialize_root_signature(root_desc);
    EXPECT(!root_blob.empty());
    Ref<ID3D12RootSignature> root;
    EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

    constexpr UINT width = 64, height = 64;
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rt_desc{};
    rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rt_desc.Width = width;
    rt_desc.Height = height;
    rt_desc.DepthOrArraySize = 1;
    rt_desc.MipLevels = 1;
    rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rt_desc.SampleDesc.Count = 1;
    rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    Ref<ID3D12Resource> target;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                     D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                     IID_PPV_ARGS(&target.p))));
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    Ref<ID3D12DescriptorHeap> rtv_heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.p, nullptr, rtv);

    // RasterizedStream 0: the ordinary case (also exercised by d3d12_m2_textured_test-style draws elsewhere, but
    // not with StreamOutput).
    EXPECT(run_case(device.p, root.p, rtv, vs_container, ps_container, width, height, 0, "RasterizedStream=0"));
    // D3D12_SO_NO_RASTERIZED_STREAM: capture-only (rasterizerDiscardEnable). Must still capture every vertex;
    // only the (unchecked) render target's contents should differ.
    EXPECT(run_case(device.p, root.p, rtv, vs_container, ps_container, width, height,
                    D3D12_SO_NO_RASTERIZED_STREAM, "NO_RASTERIZED_STREAM"));

    std::filesystem::remove_all(fixture);
    return 0;
}
