// ACE-EXPECT: detect
// CATEGORY: should_detect/tool_schema_vague
// LANGUAGE: java
// ISSUE: An OpenAI function/tool is defined with a single generic "input" string parameter and no field/tool descriptions, giving the model no signal about expected format, leading to misuse.
// EXPECTED-FINDING: Tool definition uses a bare generic parameter (string "input") with empty/missing descriptions on the tool and its properties.
// EXPECTED-FIX: Add a clear tool description and typed, well-described parameters (e.g. city: string, units: enum) so the model can call it correctly.
// SEVERITY-HINT: warning
/** Registers a weather tool with a vague single-string parameter. */
package fp.testcases;

import com.openai.core.JsonValue;
import com.openai.models.FunctionDefinition;
import com.openai.models.FunctionParameters;
import com.openai.models.chat.completions.ChatCompletionTool;

import java.util.Map;

public class JavaToolSchema1 {

    public ChatCompletionTool buildWeatherTool() {
        // Generic "input" param, no descriptions anywhere - the model has to guess the format.
        FunctionParameters params = FunctionParameters.builder()
                .putAdditionalProperty("type", JsonValue.from("object"))
                .putAdditionalProperty("properties", JsonValue.from(Map.of(
                        "input", Map.of("type", "string"))))
                .putAdditionalProperty("required", JsonValue.from(new String[]{"input"}))
                .build();

        FunctionDefinition function = FunctionDefinition.builder()
                .name("get_weather")
                .parameters(params)
                .build();

        return ChatCompletionTool.builder()
                .function(function)
                .build();
    }

    public static void main(String[] args) {
        JavaToolSchema1 app = new JavaToolSchema1();
        System.out.println(app.buildWeatherTool());
    }
}
