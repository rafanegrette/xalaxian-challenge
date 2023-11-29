package com.example.application.services;

import com.vaadin.flow.server.auth.AnonymousAllowed;
import dev.hilla.BrowserCallable;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.TokenWindowChatMemory;
import dev.langchain4j.model.embedding.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import dev.langchain4j.model.openai.OpenAiTokenizer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@BrowserCallable
@AnonymousAllowed
public class ChatService {
    public static final String GPT_3_5_TURBO = "gpt-3.5-turbo-1106";
    @Value("${openai.api.key}")
    private String OPENAI_API_KEY;

    @Value("${app.embedding.path}")
    private String EMBEDDING_PATH;
    private Assistant assistant;
    private AllMiniLmL6V2QuantizedEmbeddingModel embeddingModel;
    private InMemoryEmbeddingStore embeddingStore;
    private StreamingAssistant streamingAssistant;

    interface Assistant {
        String chat(String message);
    }

    interface StreamingAssistant {
        TokenStream chat(String message);
    }

    @PostConstruct
    public void init() {

        if (OPENAI_API_KEY == null) {
            System.err.println("ERROR: OPENAI_API_KEY environment variable is not set. Please set it to your OpenAI API key.");
        }

        var memory = TokenWindowChatMemory.withMaxTokens(4000, new OpenAiTokenizer(GPT_3_5_TURBO));

        embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();

        embeddingStore = InMemoryEmbeddingStore.fromFile(EMBEDDING_PATH);

        assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(OpenAiChatModel.builder()
                        .apiKey(OPENAI_API_KEY)
                        .modelName(GPT_3_5_TURBO)
                        .build())
                .chatMemory(memory)
                .build();

        streamingAssistant = AiServices.builder(StreamingAssistant.class)
                .streamingChatLanguageModel(OpenAiStreamingChatModel.builder()
                        .apiKey(OPENAI_API_KEY)
                        .modelName(GPT_3_5_TURBO)
                        .build())
                .chatMemory(memory)
                .build();
    }

    public String chat(String question) {
        var promptFormatted = generatePrompt(question);
        return assistant.chat(promptFormatted);
    }

    public Flux<String> chatStream(String question) {
        Sinks.Many<String> sink = Sinks.many().unicast().onBackpressureBuffer();

        var promptFormatted = generatePrompt(question);
        streamingAssistant.chat(promptFormatted)
                .onNext(sink::tryEmitNext)
                //.onComplete(sink::tryEmitComplete)
                .onError(sink::tryEmitError)
                .start();

        return sink.asFlux();
    }

    private String generatePrompt(String question) {
        var questionEmbedding = embeddingModel.embed(question).content();
        int maxResults = 5;
        double minScore = 0.8;

        List<EmbeddingMatch<TextSegment>> relevantEmbeddings = embeddingStore.findRelevant(questionEmbedding, maxResults, minScore);

        PromptTemplate promptTemplate = PromptTemplate.from(
                """
                        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know. Use three sentences maximum. Remember you are an AI created by an Alien race called Xalaxian, to help humans to know them, you are very friendly.
                        
                        {{context}}
                        
                        Question: {{question}}
                        Helpful Answer:
                        """
        );

        String context = relevantEmbeddings.stream().map(match -> match.embedded().text()).collect(Collectors.joining("\n"));

        Map<String, Object> variables = new HashMap<>();
        variables.put("question", question);
        variables.put("context", context);

        Prompt prompt = promptTemplate.apply(variables);
        return prompt.toUserMessage().text();
    }
}
