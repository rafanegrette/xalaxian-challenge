package com.example.application.services;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentType;
import dev.langchain4j.data.document.FileSystemDocumentLoader;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiTokenizer;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStoreJsonCodec;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import static dev.langchain4j.data.document.FileSystemDocumentLoader.loadDocument;
import static dev.langchain4j.data.document.FileSystemDocumentLoader.loadDocuments;
import static dev.langchain4j.model.openai.OpenAiModelName.GPT_3_5_TURBO;
import static org.junit.jupiter.api.Assertions.*;


class ChatServiceTest {

    //@Test
    void uploadData() {
        String path = "C:\\Users\\rafael.negrette\\Downloads\\dataset-20231101T200248Z-001\\dataset";
        List<Document> documents = getTextDocuments(path);

        var splitter = DocumentSplitters.recursive(150, 10, new OpenAiTokenizer("gpt-3.5-turbo-1106"));
        var segments = splitter.splitAll(documents);

        var embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();
        var embeddings = embeddingModel.embedAll(segments).content();

        InMemoryEmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        var embeddingPath = "C:\\Users\\rafael.negrette\\Models\\Embeddings\\xalaxian-embeddings.store";
        embeddingStore.serializeToFile(embeddingPath);

        assertEquals(5, segments.size());
    }

    //@Test
    void chat() {
    }

    private List<Document> getTextDocuments(String pathStr) {
        var folderPath = toPath(pathStr);
        List<Document> documents = new ArrayList<>();
        try {
             documents = Files.list(folderPath).filter(p -> {
                var fileName = p.toString();
                return fileName.endsWith(".pdf") || fileName.endsWith(".txt") || fileName.endsWith(".docx");
            }).map(path -> {
                Document document = null;
                try {
                    document = loadDocument(path);
                } catch (Exception e) {
                    try {
                        document = loadDocument(path, DocumentType.TXT);
                    } catch (Exception innerExcep) {
                        System.err.println("Double Error Loading: " + path);
                    }
                    System.err.println("Error Loading: " + path);
                }
                return document;
             }).collect(Collectors.toList());
        } catch (IOException e) {
            System.out.println("Error reading files");
        }
        return documents;

    }
    private static Path toPath(String fileName) {
            return Paths.get(fileName);

    }
}