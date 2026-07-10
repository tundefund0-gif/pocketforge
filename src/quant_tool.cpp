#include "forge/gguf_reader.hpp"
#include "forge/quant_format.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <unordered_map>

using namespace forge;

// ============================================================
//  Tensor name patterns for model architectures
// ============================================================
// Supports multiple naming conventions:
//   llama.cpp:   blk.{l}.attn_{q,k,v,output}.weight
//   HF convert:  model.layers.{l}.self_attn.{q,k,v,o}_proj.weight
//   mlp:         {gate,up,down}_proj.weight or ffn_{gate,up,down}.weight
// ============================================================

static std::string tensor_name_q(const GGUFFile& gguf, uint32_t l) {
    std::vector<std::string> candidates = {
        "blk." + std::to_string(l) + ".attn_q.weight",
        "model.layers." + std::to_string(l) + ".self_attn.q_proj.weight",
    };
    for (auto& c : candidates) {
        if (gguf.tensor_info(c)) return c;
    }
    return "";
}

static std::string tensor_name_k(const GGUFFile& gguf, uint32_t l) {
    std::vector<std::string> candidates = {
        "blk." + std::to_string(l) + ".attn_k.weight",
        "model.layers." + std::to_string(l) + ".self_attn.k_proj.weight",
    };
    for (auto& c : candidates) {
        if (gguf.tensor_info(c)) return c;
    }
    return "";
}

static std::string tensor_name_v(const GGUFFile& gguf, uint32_t l) {
    std::vector<std::string> candidates = {
        "blk." + std::to_string(l) + ".attn_v.weight",
        "model.layers." + std::to_string(l) + ".self_attn.v_proj.weight",
    };
    for (auto& c : candidates) {
        if (gguf.tensor_info(c)) return c;
    }
    return "";
}

static std::string tensor_name_o(const GGUFFile& gguf, uint32_t l) {
    std::vector<std::string> candidates = {
        "blk." + std::to_string(l) + ".attn_output.weight",
        "model.layers." + std::to_string(l) + ".self_attn.o_proj.weight",
    };
    for (auto& c : candidates) {
        if (gguf.tensor_info(c)) return c;
    }
    return "";
}

static std::string tensor_name_gate(const GGUFFile& gguf, uint32_t l) {
    std::vector<std::string> candidates = {
        "blk." + std::to_string(l) + ".ffn_gate.weight",
        "model.layers." + std::to_string(l) + ".mlp.gate_proj.weight",
    };
    for (auto& c : candidates) {
        if (gguf.tensor_info(c)) return c;
    }
    return "";
}

static std::string tensor_name_up(const GGUFFile& gguf, uint32_t l) {
    std::vector<std::string> candidates = {
        "blk." + std::to_string(l) + ".ffn_up.weight",
        "model.layers." + std::to_string(l) + ".mlp.up_proj.weight",
    };
    for (auto& c : candidates) {
        if (gguf.tensor_info(c)) return c;
    }
    return "";
}

static std::string tensor_name_down(const GGUFFile& gguf, uint32_t l) {
    std::vector<std::string> candidates = {
        "blk." + std::to_string(l) + ".ffn_down.weight",
        "model.layers." + std::to_string(l) + ".mlp.down_proj.weight",
    };
    for (auto& c : candidates) {
        if (gguf.tensor_info(c)) return c;
    }
    return "";
}

static std::string tensor_name_embed(const GGUFFile& gguf) {
    std::vector<std::string> candidates = {
        "token_embd.weight",
        "model.embed_tokens.weight",
        "gpt_neox.embed_in.weight",
    };
    for (auto& c : candidates) {
        if (gguf.tensor_info(c)) return c;
    }
    return "";
}

static std::string tensor_name_output(const GGUFFile& gguf) {
    std::vector<std::string> candidates = {
        "output.weight",
        "lm_head.weight",
    };
    for (auto& c : candidates) {
        if (gguf.tensor_info(c)) return c;
    }
    return "";
}

static std::string tensor_name_attn_norm(const GGUFFile& gguf, uint32_t l) {
    std::vector<std::string> candidates = {
        "blk." + std::to_string(l) + ".attn_norm.weight",
        "model.layers." + std::to_string(l) + ".input_layernorm.weight",
    };
    for (auto& c : candidates) {
        if (gguf.tensor_info(c)) return c;
    }
    return "";
}

static std::string tensor_name_ffn_norm(const GGUFFile& gguf, uint32_t l) {
    std::vector<std::string> candidates = {
        "blk." + std::to_string(l) + ".ffn_norm.weight",
        "model.layers." + std::to_string(l) + ".post_attention_layernorm.weight",
    };
    for (auto& c : candidates) {
        if (gguf.tensor_info(c)) return c;
    }
    return "";
}

static std::string tensor_name_final_norm(const GGUFFile& gguf) {
    std::vector<std::string> candidates = {
        "output_norm.weight",
        "model.norm.weight",
        "gpt_neox.final_layer_norm.weight",
    };
    for (auto& c : candidates) {
        if (gguf.tensor_info(c)) return c;
    }
    return "";
}

// ============================================================
//  Main quantizer tool
// ============================================================

void print_usage() {
    std::cerr << "Usage: forge-quant <input.gguf> <output.squeeze> [options]\n";
    std::cerr << "Options:\n";
    std::cerr << "  --list-tensors         List all tensor names and exit\n";
    std::cerr << "  --quant-type TYPE      Quantization type: q8, q4, q2, q1.5 (default: q8)\n";
    std::cerr << "  --embed-quant TYPE     Embedding quantization type (default: q8)\n";
    std::cerr << "  --layers N            Override layer count\n";
    std::cerr << "  --embd N              Override embedding dimension\n";
    std::cerr << "  --heads N             Override head count\n";
    std::cerr << "  --kv-heads N          Override KV head count\n";
    std::cerr << "  --ff N                Override FFN dimension\n";
    std::cerr << "  --vocab N             Override vocabulary size\n";
    std::cerr << "  --max-ctx N           Override max context length\n";
    std::cerr << "  --mtp N               MTP heads (default: 4)\n";
}

static uint8_t parse_quant_type(const std::string& s) {
    if (s == "q8" || s == "Q8") return 0;
    if (s == "q4" || s == "Q4") return 1;
    if (s == "q2" || s == "Q2") return 2;
    if (s == "q1.5" || s == "Q1.5" || s == "ternary") return 3;
    std::cerr << "Unknown quant type '" << s << "', using Q8\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    bool list_tensors_only = false;
    std::string quant_type_str = "q8";
    std::string embed_quant_str = "q8";
    ModelConfig cfg_override;
    bool has_cfg_override = false;

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--list-tensors") list_tensors_only = true;
        else if (arg == "--quant-type" && i + 1 < argc) quant_type_str = argv[++i];
        else if (arg == "--embed-quant" && i + 1 < argc) embed_quant_str = argv[++i];
        else if (arg == "--layers" && i + 1 < argc) { cfg_override.n_layers = (uint32_t)std::atoi(argv[++i]); has_cfg_override = true; }
        else if (arg == "--embd" && i + 1 < argc) { cfg_override.n_embd = (uint32_t)std::atoi(argv[++i]); has_cfg_override = true; }
        else if (arg == "--heads" && i + 1 < argc) { cfg_override.n_heads = (uint32_t)std::atoi(argv[++i]); has_cfg_override = true; }
        else if (arg == "--kv-heads" && i + 1 < argc) { cfg_override.n_kv_heads = (uint32_t)std::atoi(argv[++i]); has_cfg_override = true; }
        else if (arg == "--ff" && i + 1 < argc) { cfg_override.n_ff = (uint32_t)std::atoi(argv[++i]); has_cfg_override = true; }
        else if (arg == "--vocab" && i + 1 < argc) { cfg_override.n_vocab = (uint32_t)std::atoi(argv[++i]); has_cfg_override = true; }
        else if (arg == "--max-ctx" && i + 1 < argc) { cfg_override.max_seq_len = (uint32_t)std::atoi(argv[++i]); has_cfg_override = true; }
        else if (arg == "--mtp" && i + 1 < argc) { cfg_override.mtp_heads = (uint32_t)std::atoi(argv[++i]); }
    }

    uint8_t quant_type = parse_quant_type(quant_type_str);
    uint8_t embed_quant = parse_quant_type(embed_quant_str);

    std::cout << "=== PocketForge Quantizer ===\n";
    std::cout << "Input:  " << input_path << "\n";
    std::cout << "Output: " << output_path << "\n";

    // ============================================================
    //  Open GGUF file
    // ============================================================
    GGUFFile gguf;
    if (!gguf.open(input_path)) {
        std::cerr << "Failed to open GGUF file: " << input_path << "\n";
        return 1;
    }

    // Read config from GGUF
    ModelConfig cfg;
    if (has_cfg_override) {
        cfg = cfg_override;
        auto gguf_cfg = gguf.read_config();
        if (cfg.n_layers == 0) cfg.n_layers = gguf_cfg.n_layers;
        if (cfg.n_embd == 0) cfg.n_embd = gguf_cfg.n_embd;
        if (cfg.n_heads == 0) cfg.n_heads = gguf_cfg.n_heads;
        if (cfg.n_kv_heads == 0) cfg.n_kv_heads = gguf_cfg.n_kv_heads;
        if (cfg.n_ff == 0) cfg.n_ff = gguf_cfg.n_ff;
        if (cfg.n_vocab == 0) cfg.n_vocab = gguf_cfg.n_vocab;
        if (cfg.max_seq_len == 131072) cfg.max_seq_len = gguf_cfg.max_seq_len;
    } else {
        cfg = gguf.read_config();
    }
    cfg.mtp_heads = cfg_override.mtp_heads;

    std::cout << "Model: " << cfg.n_layers << " layers, " << cfg.n_embd << " embd, "
              << cfg.n_heads << " heads (" << cfg.n_kv_heads << " KV), "
              << cfg.n_ff << " FF, " << cfg.n_vocab << " vocab\n";
    std::cout << "Max context: " << cfg.max_seq_len << "\n";
    std::cout << "Quant type: " << quant_type_str << "\n";
    std::cout << "MTP heads: " << cfg.mtp_heads << "\n";

    // ============================================================
    //  List tensors (for debugging)
    // ============================================================
    auto all_tensors = gguf.list_tensors();
    std::cout << "\nFound " << all_tensors.size() << " tensors:\n";
    for (const auto& name : all_tensors) {
        auto* info = gguf.tensor_info(name);
        if (info) {
            std::cout << "  " << name << " [" << info->n_dims << "D";
            for (auto d : info->dims) std::cout << "x" << d;
            std::cout << " type=" << info->type << "]\n";
        }
    }

    if (list_tensors_only) {
        gguf.close();
        return 0;
    }

    // ============================================================
    //  Quantize and write .squeeze file
    // ============================================================
    Quantizer quantizer(cfg);
    std::vector<WeightBlock> blocks;
    std::vector<std::vector<uint8_t>> compressed_blocks;

    const char* matrix_names[] = {"Q", "K", "V", "O", "gate", "up", "down"};

    // ============================================================
    //  Layer weights (7 matrices per layer)
    // ============================================================
    for (uint32_t l = 0; l < cfg.n_layers; l++) {
        std::cout << "\nLayer " << l << "...\n";

        std::string names[7] = {
            tensor_name_q(gguf, l),
            tensor_name_k(gguf, l),
            tensor_name_v(gguf, l),
            tensor_name_o(gguf, l),
            tensor_name_gate(gguf, l),
            tensor_name_up(gguf, l),
            tensor_name_down(gguf, l)
        };

        for (uint32_t m = 0; m < 7; m++) {
            if (names[m].empty()) {
                std::cerr << "  WARNING: Cannot find tensor for layer " << l << " matrix " << matrix_names[m] << "\n";
                continue;
            }

            std::vector<float> weights = gguf.dequantize_tensor(names[m]);
            if (weights.empty()) {
                std::cerr << "  WARNING: Failed to dequantize " << names[m] << "\n";
                continue;
            }

            auto* info = gguf.tensor_info(names[m]);
            uint32_t n_rows = info ? (uint32_t)info->dims[1] : cfg.n_embd;
            uint32_t n_cols = info ? (uint32_t)info->dims[0] : cfg.n_embd;

            auto quantized = quantizer.quantize_matrix(weights.data(), n_rows, n_cols, quant_type);
            if (quantized.empty()) {
                std::cerr << "  WARNING: Quantization failed for " << names[m] << "\n";
                continue;
            }

            auto compressed = quantizer.compress_block(quantized);

            WeightBlock block;
            block.layer_id = l;
            block.matrix_id = m;
            block.n_rows = n_rows;
            block.n_cols = n_cols;
            block.compressed_size = (uint32_t)compressed.size();
            block.original_size = (uint32_t)quantized.size();
            block.quant_type = quant_type;
            block.offset = 0;

            std::cout << "  " << matrix_names[m] << " [" << n_rows << "x" << n_cols << "] "
                      << "-> " << (quant_type == 0 ? "Q8" : quant_type == 1 ? "Q4" :
                                  quant_type == 2 ? "Q2" : "Q1.5")
                      << " (" << compressed.size() << " bytes)\n";

            blocks.push_back(block);
            compressed_blocks.push_back(std::move(compressed));
        }
    }

    // ============================================================
    //  Embedding table
    // ============================================================
    {
        std::string emb_name = tensor_name_embed(gguf);
        if (!emb_name.empty()) {
            std::cout << "\nEmbedding table: " << emb_name << "\n";
            std::vector<float> emb_weights = gguf.dequantize_tensor(emb_name);
            if (!emb_weights.empty()) {
                auto* info = gguf.tensor_info(emb_name);
                uint32_t n_rows = info ? (uint32_t)info->dims[1] : cfg.n_vocab;
                uint32_t n_cols = info ? (uint32_t)info->dims[0] : cfg.n_embd;

                auto quantized = quantizer.quantize_matrix(emb_weights.data(), n_rows, n_cols, embed_quant);
                auto compressed = quantizer.compress_block(quantized);

                WeightBlock block;
                block.layer_id = 0;
                block.matrix_id = 10; // special ID for embedding
                block.n_rows = n_rows;
                block.n_cols = n_cols;
                block.compressed_size = (uint32_t)compressed.size();
                block.original_size = (uint32_t)quantized.size();
                block.quant_type = embed_quant;
                block.offset = 0;

                std::cout << "  Embed [" << n_rows << "x" << n_cols << "] "
                          << "-> " << (embed_quant == 0 ? "Q8" : embed_quant == 1 ? "Q4" :
                                      embed_quant == 2 ? "Q2" : "Q1.5")
                          << " (" << compressed.size() << " bytes)\n";

                blocks.push_back(block);
                compressed_blocks.push_back(std::move(compressed));
            }
        }
    }

    // ============================================================
    //  Output projection (lm_head) if separate
    // ============================================================
    {
        std::string out_name = tensor_name_output(gguf);
        if (!out_name.empty()) {
            std::cout << "\nOutput projection: " << out_name << "\n";
            std::vector<float> out_weights = gguf.dequantize_tensor(out_name);
            if (!out_weights.empty()) {
                auto* info = gguf.tensor_info(out_name);
                uint32_t n_rows = info ? (uint32_t)info->dims[1] : cfg.n_vocab;
                uint32_t n_cols = info ? (uint32_t)info->dims[0] : cfg.n_embd;

                auto quantized = quantizer.quantize_matrix(out_weights.data(), n_rows, n_cols, quant_type);
                auto compressed = quantizer.compress_block(quantized);

                WeightBlock block;
                block.layer_id = 0;
                block.matrix_id = 11; // special ID for output projection
                block.n_rows = n_rows;
                block.n_cols = n_cols;
                block.compressed_size = (uint32_t)compressed.size();
                block.original_size = (uint32_t)quantized.size();
                block.quant_type = quant_type;
                block.offset = 0;

                std::cout << "  Output [" << n_rows << "x" << n_cols << "] "
                          << "-> " << (quant_type == 0 ? "Q8" : quant_type == 1 ? "Q4" :
                                      quant_type == 2 ? "Q2" : "Q1.5")
                          << " (" << compressed.size() << " bytes)\n";

                blocks.push_back(block);
                compressed_blocks.push_back(std::move(compressed));
            }
        }
    }

    // ============================================================
    //  Write .squeeze file
    // ============================================================
    if (quantizer.write_squeeze(output_path, blocks, compressed_blocks, cfg)) {
        std::cout << "\nOK Wrote " << output_path << "\n";
        std::cout << "  " << blocks.size() << " weight blocks\n";

        size_t total = 0;
        for (const auto& cb : compressed_blocks) total += cb.size();
        std::cout << "  Total compressed: " << total / (1024 * 1024) << " MB\n";
        std::cout << "  Estimated RAM usage during inference: ~"
                  << (total / (1024 * 1024) / 2) << " MB (mmap'd)\n";
        gguf.close();
        return 0;
    } else {
        std::cerr << "Failed to write " << output_path << "\n";
        gguf.close();
        return 1;
    }
}
