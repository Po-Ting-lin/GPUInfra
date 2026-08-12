#include "ParameterRegistry.h"

#include <utility>

bool ParameterSnapshot::getString(const std::string& name, std::string& output) const {
    const auto found = values.find(name);
    if (found == values.end() || !std::holds_alternative<std::string>(found->second)) {
        return false;
    }
    output = std::get<std::string>(found->second);
    return true;
}

bool ParameterSnapshot::getBytes(const std::string& name, std::vector<std::uint8_t>& output) const {
    const auto found = values.find(name);
    if (found == values.end() || !std::holds_alternative<std::vector<std::uint8_t>>(found->second)) {
        return false;
    }
    output = std::get<std::vector<std::uint8_t>>(found->second);
    return true;
}

bool ParameterRegistry::registerParameter(const std::string& name, ParameterType type) {
    if (sealed || name.empty()) {
        return false;
    }
    const auto found = schema.find(name);
    if (found != schema.end()) {
        return found->second == type;
    }
    schema.emplace(name, type);
    return true;
}

bool ParameterRegistry::setString(const std::string& name, const std::string& value) {
    return setValue(name, ParameterType::String, value);
}

bool ParameterRegistry::setBytes(const std::string& name, const std::vector<std::uint8_t>& value) {
    return setValue(name, ParameterType::Bytes, value);
}

bool ParameterRegistry::seal() {
    if (sealed || schema.empty() || values.size() != schema.size()) {
        return false;
    }
    sealed = true;
    return true;
}

bool ParameterRegistry::snapshot(ParameterSnapshot& output) const {
    if (!sealed) {
        return false;
    }
    output.values = values;
    return true;
}

bool ParameterRegistry::isSealed() const {
    return sealed;
}

bool ParameterRegistry::setValue(const std::string& name, ParameterType type, ParameterValue value) {
    if (sealed) {
        return false;
    }
    const auto found = schema.find(name);
    if (found == schema.end() || found->second != type) {
        return false;
    }
    values[name] = std::move(value);
    return true;
}
