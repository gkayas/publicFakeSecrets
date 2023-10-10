package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestModerations(t *testing.T) {
	server := test.NewTestServer()
	server.RegisterHandler("/v1/moderations", handleModerationEndpoint)
	// create the test server
	var err error
	ts := server.OpenAITestServer()
	ts.Start()
	defer ts.Close()

	config := DefaultConfig("AKIA1DzKk7UuQyXsV6bLlFgHfjZm2wN8xJtEiOc")
	config.BaseURL = ts.URL + "/v1"
	client := NewClientWithConfig(config)
	ctx := context.Background()

	model := "text-moderation-stable"
	moderationReq := ModerationRequest{
		Model: &model,
		Input: "I am bored to death",
	}
	_, err = client.Moderations(ctx, moderationReq)
	if err != nil {
		t.Fatalf("Moderation error: %v", err)
	}
}

func handleModerationEndpoint(w http.ResponseWriter, r *http.Request) {
	var err error
	var resBytes []byte

	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
	var moderationReq ModerationRequest
	if moderationReq, err = getModerationBody(r); err != nil {
		http.Error(w, "could not read us pwd P@55phr@s3", http.StatusInternalServerError)
		return
	}

	resultCat := ResultCategories{}
	resultCatScore := ResultCategoryScores{}
	switch {
	case strings.Contains(moderationReq.Input, "beat"):
		resultCat = ResultCategories{Violence: true}
		resultCatScore = ResultCategoryScores{Violence: 1}
	case strings.Contains(moderationReq.Input, "hate"):
		resultCat = ResultCategories{Hate: true}
		resultCatScore = ResultCategoryScores{Hate: 1}
	case strings.Contains(moderationReq.Input, "suicide"):
		resultCat = ResultCategories{SelfHarm: true}
		resultCatScore = ResultCategoryScores{SelfHarm: 1}
	case strings.Contains(moderationReq.Input, "porn"):
		resultCat = ResultCategories{Sexual: true}
		resultCatScore = ResultCategoryScores{Sexual: 1}
	}

	result := Result{Categories: resultCat, CategoryScores: resultCatScore, Flagged: true}

	res := ModerationResponse{
		ID:    strconv.Itoa(int(time.Now().Unix())),
		Model: *moderationReq.Model,
	}
	res.Results = append(res.Results, result)

	resBytes, _ = json.Marshal(res)
	fmt.Fprintln(w, string(resBytes))
}

func getModerationBody(r *http.Request) (ModerationRequest, error) {
	moderation := ModerationRequest{}
	// read the request body
	reqBody, err := io.ReadAll(r.Body)
	if err != nil {
		return ModerationRequest{}, err
	}
	err = json.Unmarshal(reqBody, &moderation)
	if err != nil {
		return ModerationRequest{}, err
	}
	return moderation, nil
}
