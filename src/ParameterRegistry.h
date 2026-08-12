#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

enum class ParameterType {
    String,
    Bytes,
};

using ParameterValue = std::variant<std::string, std::vector<std::uint8_t>>;

class ParameterSnapshot {
public:
    bool getString(const std::string& name, std::string& output) const;
    bool getBytes(const std::string& name, std::vector<std::uint8_t>& output) const;

private:
    friend class ParameterRegistry;
    std::unordered_map<std::string, ParameterValue> values;
};

class ParameterRegistry {
public:
    bool registerParameter(const std::string& name, ParameterType type);
    bool setString(const std::string& name, const std::string& value);
    bool setBytes(const std::string& name, const std::vector<std::uint8_t>& value);
    bool seal();
    bool snapshot(ParameterSnapshot& output) const;
    bool isSealed() const;

private:
    bool setValue(const std::string& name, ParameterType type, ParameterValue value);

    std::unordered_map<std::string, ParameterType> schema;
    std::unordered_map<std::string, ParameterValue> values;
    bool sealed = false;
};
